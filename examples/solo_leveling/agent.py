import json
import random
import statistics
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evogym.utils import is_connected, has_actuator, get_full_connectivity, hashable

from solo_leveling.config import (
    SKILL_WEIGHT_DELTA_CAP,
)
from solo_leveling.json_io import dump_json
from utils.algo_utils import mutate, compute_voxel_diff
from solo_leveling.llm_client import call_llm
from solo_leveling.skill_library import (
    SkillLibrary,
    _format_l2_rules_for_llm,
    _is_frozen,
    _skill_condition,
    _skill_positive_rules,
    _skill_negative_rules,
)
from solo_leveling.prompts import (
    PROPOSE_INIT_PROMPT, PROPOSE_MUTATE_PROMPT,
    UNRETRIEVED_DIAGNOSE_PROMPT, VOXEL_LEGEND,
    _TC_SAME_GRID_BASE, _TC_CROSS_GRID_BASE,
    _TC_SAME_GRID_ELITE_ADDENDUM, _TC_CROSS_GRID_ELITE_ADDENDUM,
)
from solo_leveling.evaluate import evaluate_batch
from solo_leveling.new_insight_v2 import run_new_insight
from solo_leveling.pipeline_v2 import run_one_generation


TASK_DESCRIPTIONS = {
    # ===== 5×5 Baseline =====
    "Walker-v0": (
        "Walker-v0: A soft robot (5×5 voxel grid) on flat ground under gravity. "
        "Starts stationary. "
        "Reward per step = horizontal displacement of the robot's center of mass (rightward positive). "
        "Bonus: +1.0 and termination when the robot's center of mass crosses the right end of the floor. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Carrier-v0": (
        "Carrier-v0: A soft robot (5×5 voxel grid) on flat ground under gravity. "
        "A 3-wide × 2-tall soft box rests directly on top of the robot at the start. "
        "The robot must transport the box rightward across the floor without dropping it. "
        "Reward per step = 0.5 × box horizontal displacement + 0.5 × robot horizontal displacement (both rightward positive). "
        "Penalty: when the box center of mass drops below a height threshold (3 voxels), "
        "every subsequent step adds 10 × (box height change) — "
        "falling further is penalized, lifting the box back up is rewarded at the same rate. "
        "Bonus: +1.0 and termination when the robot's center of mass crosses the right end of the floor. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Pusher-v0": (
        "Pusher-v0: A soft robot (5×5 voxel grid) on flat ground under gravity. "
        "A 3-wide × 2-tall soft box sits at floor level a few voxels to the right of the robot's starting position. "
        "The robot must push the box rightward across the floor without losing contact with it. "
        "Reward per step = 0.75 × box horizontal displacement + 0.5 × robot horizontal displacement "
        "+ (decrease in box-robot horizontal separation, separation increase is penalized). "
        "Bonus: +1.0 and termination when the robot's center of mass crosses the right end of the floor. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Catcher-v0": (
        "Catcher-v0: A soft robot on flat ground under gravity. "
        "A 3-wide × 2-tall soft package spawns from above, with two rigid pegs positioned to deflect its falling trajectory. "
        "Spawn positions are randomized each episode within a small window, so the package's path is unpredictable. "
        "The robot must position itself horizontally to catch the package as it falls. "
        "Reward per step = decrease in horizontal distance between robot and package centers of mass. "
        "Penalty: when the package falls below a height threshold (5 voxels), "
        "every subsequent step adds 10 × (package height change). "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "BridgeWalker-v0": (
        "BridgeWalker-v0: A soft robot (5×5 voxel grid) standing on a deformable bridge under gravity. "
        "The bridge consists of single-cell rigid pillars supporting alternating soft spans between them; "
        "the soft spans sag under the robot's weight, requiring it to maintain balance while traversing. "
        "Reward per step = horizontal displacement of the robot's center of mass (rightward positive). "
        "Bonus: +1.0 and termination when the robot's center of mass crosses past the right end of the bridge. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "UpStepper-v0": (
        "UpStepper-v0: A soft robot (5×5 voxel grid) on a rising staircase terrain under gravity. "
        "The robot must travel rightward while climbing a sequence of upward steps. "
        "Reward per step = horizontal displacement of the robot's center of mass (rightward positive). "
        "Bonus: +2.0 when the robot's center of mass reaches the far right end of the course (successful traversal). "
        "Penalty: -3.0 and early termination if the robot rotates roughly 90° or more from upright (tips over). "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Climber-v0": (
        "Climber-v0: A soft robot (5×5 voxel grid) inside a vertical climbing chute under gravity. "
        "The chute is bounded by rigid walls on both sides at exactly the robot's width — "
        "the robot must press against and grip the walls to ascend, since there is no floor support past the bottom. "
        "Reward per step = vertical displacement of the robot's center of mass (upward positive). "
        "Bonus: +1.0 and termination when the robot's center of mass reaches near the top of the chute. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Thrower-v0": (
        "Thrower-v0: A soft robot (5×5 voxel grid) on flat ground under gravity. "
        "A 3-wide × 2-tall soft package drops onto the robot from above at the start of the episode. "
        "The goal is to launch the package as far to the right as possible. "
        "Reward per step = (package horizontal displacement, rightward positive) + 0.25 × (robot's leftward displacement). "
        "The second term discourages the robot from chasing the package rightward — the robot is "
        "rewarded for staying put or recoiling slightly leftward while propelling the package. "
        "However, if the robot's center of mass drifts past the world's left boundary, the sign of "
        "that term flips, penalizing further leftward motion. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Jumper-v0": (
        "Jumper-v0: A soft robot (5×5 voxel grid) on flat ground under gravity. "
        "The robot starts stationary and must jump upward while minimizing step-to-step horizontal motion. "
        "Reward per step = 10 × vertical displacement of the robot's center of mass "
        "- 5 × absolute horizontal displacement of the robot's center of mass over that step. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Balancer-v0": (
        "Balancer-v0: A soft robot (5×5 voxel grid) under gravity, initialized on a narrow balancing post. "
        "The post is a 1-wide × 2-tall rigid support above the floor. "
        "The robot must keep its center of mass near a fixed balance point above the post; "
        "a centered body starts approximately aligned with this target, so reward comes from resisting drift and recovering balance. "
        "Reward per step = decrease in absolute horizontal distance to the target x-position "
        "+ decrease in absolute vertical distance to the target y-position. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),

    # ===== 10×10 Variants (self-contained, no cross-reference to 5×5) =====
    "Walker-v0-10x10": (
        "Walker-v0-10x10: A soft robot (10×10 voxel grid) on a long flat track under gravity. "
        "Starts stationary. "
        "Reward per step = horizontal displacement of the robot's center of mass (rightward positive). "
        "Bonus: +1.0 and termination when the robot's center of mass crosses the right end of the track. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Pusher-v0-10x10": (
        "Pusher-v0-10x10: A soft robot (10×10 voxel grid) on a flat track under gravity. "
        "A 6-wide × 4-tall soft box sits at floor level a few voxels to the right of the robot's starting position. "
        "The robot must push the box rightward across the track without losing contact with it. "
        "Reward per step = 0.75 × box horizontal displacement + 0.5 × robot horizontal displacement "
        "+ (decrease in box-robot horizontal separation, separation increase is penalized). "
        "Bonus: +1.0 and termination when the robot's center of mass crosses the right end of the track. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Carrier-v0-10x10": (
        "Carrier-v0-10x10: A soft robot (10×10 voxel grid) on a flat track under gravity. "
        "A 6-wide × 4-tall soft box rests directly on top of the robot at the start. "
        "The robot must transport the box rightward across the track without dropping it. "
        "Reward per step = 0.5 × box horizontal displacement + 0.5 × robot horizontal displacement (both rightward positive). "
        "Penalty: when the box center of mass drops below a height threshold (6 voxels), "
        "every subsequent step adds 10 × (box height change) — "
        "falling further is penalized, lifting the box back up is rewarded at the same rate. "
        "Bonus: +1.0 and termination when the robot's center of mass crosses the right end of the track. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "BridgeWalker-v0-10x10": (
        "BridgeWalker-v0-10x10: A soft robot (10×10 voxel grid) standing on a deformable bridge under gravity. "
        "The bridge consists of single-cell rigid pillars supporting alternating soft spans between them; "
        "the soft spans sag noticeably under the robot's weight, requiring it to maintain balance while traversing. "
        "Reward per step = horizontal displacement of the robot's center of mass (rightward positive). "
        "Bonus: +1.0 and termination when the robot's center of mass crosses past the right end of the bridge. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "UpStepper-v0-10x10": (
        "UpStepper-v0-10x10: A soft robot (10×10 voxel grid) on a rising staircase terrain under gravity. "
        "Each stair plateau is wide enough for the robot to fit on a single step level. "
        "The robot must travel rightward while climbing a sequence of upward steps. "
        "Reward per step = horizontal displacement of the robot's center of mass (rightward positive). "
        "Bonus: +2.0 when the robot's center of mass reaches the far right end of the course (successful traversal). "
        "Penalty: -3.0 and early termination if the robot rotates roughly 90° or more from upright (tips over). "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Climber-v0-10x10": (
        "Climber-v0-10x10: A soft robot (10×10 voxel grid) inside a vertical climbing chute under gravity. "
        "The chute is bounded by rigid walls on both sides at exactly the robot's width — "
        "the robot must press against and grip the walls to ascend, since there is no floor support past the bottom. "
        "Reward per step = vertical displacement of the robot's center of mass (upward positive). "
        "Bonus: +1.0 and termination when the robot's center of mass reaches near the top of the chute. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Jumper-v0-10x10": (
        "Jumper-v0-10x10: A soft robot (10×10 voxel grid) on a scaled flat-ground jumping arena under gravity. "
        "The robot starts stationary and must jump upward while minimizing step-to-step horizontal motion. "
        "Reward per step = 10 × vertical displacement of the robot's center of mass "
        "- 5 × absolute horizontal displacement of the robot's center of mass over that step. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
    "Balancer-v0-10x10": (
        "Balancer-v0-10x10: A soft robot (10×10 voxel grid) under gravity, initialized on a scaled balancing post. "
        "The post is a 2-wide × 4-tall rigid support above the floor. "
        "The robot must use its larger body to keep its center of mass near a fixed balance point above the post; "
        "a centered body starts approximately aligned with this target, so reward comes from resisting drift and recovering balance. "
        "Reward per step = decrease in absolute horizontal distance to the target x-position "
        "+ decrease in absolute vertical distance to the target y-position. "
        "Simulation terminates early with a -3.0 penalty if physics becomes unstable."
    ),
}

class SoloLevelingAgent:
    def __init__(
        self,
        env_name: str,
        exp_name: str,
        save_root: str,
        config: Namespace,
        init_skill_library: str | None = None,
        init_designs_path: str | None = None,
        retrieve_as_task: str | None = None,
        elite_inject: bool = True,
        n_static_elites: int = 5,
        static_active_gens: int = 5,
    ):
        self.env_name = env_name
        self.exp_name = exp_name
        self.save_root = Path(save_root)
        self.config = config

        # Skills are per-experiment to avoid cross-contamination between runs.
        # To share skills across experiments, symlink or copy the skills/ directory.
        self.skills_dir = self.save_root / "logs" / exp_name / "skills"
        self.logs_dir = self.save_root / "logs" / exp_name
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Static prior import (warm-start). Returns the effective static_active_gens
        # (None if no skill was actually imported = cold-start fallback).
        self.static_elites = []
        if init_skill_library:
            self._effective_static_active_gens = self._import_static_payload(
                src_skills_dir=Path(init_skill_library),
                retrieve_as_task=retrieve_as_task,
                elite_inject=elite_inject,
                n_static_elites=n_static_elites,
                static_active_gens=static_active_gens,
            )
        else:
            self._effective_static_active_gens = None

        self._elite_inject_enabled = elite_inject

        self.skill_library = SkillLibrary(
            str(self.skills_dir),
            static_active_gens=self._effective_static_active_gens,
        )

        self.elites: list[dict] = []  # {"body": list, "fitness": float, "generation": int}
        self._label_counter = 0
        # Cross-generation dedup: hash -> fitness (reuse past evaluations)
        self._seen_designs: dict[str, float] = {}
        self._eval_count = 0  # total PPO evaluations consumed so far
        self._parent_history_index: dict[str, list[dict]] = {}
        self._body_by_hash: dict[str, list[list[int]]] = {}
        self._rebuild_parent_history_index()

        # Optional Gen 0 fixed init population (loaded from a GA baseline output)
        if init_designs_path:
            self._init_designs = self._load_init_population(Path(init_designs_path))
            self._eval_count = len(self._init_designs)
            print(f"  [Init] Loaded {len(self._init_designs)} pre-evaluated init designs "
                  f"from {init_designs_path}")
        else:
            self._init_designs = None

    def _infer_source_grid(self, src_skills_dir: Path) -> tuple[int, int]:
        """Infer source experiment's voxel grid shape from its data.

        Priority: source skill L3 body -> state_snapshot elites -> designs.json
        -> fall back to current STRUCTURE_SHAPE (with warning).
        """
        def _shape_from_body(body) -> tuple[int, int] | None:
            if not isinstance(body, list) or not body:
                return None
            first = body[0]
            if not isinstance(first, list) or not first:
                return None
            H = len(body)
            W = len(first)
            if H <= 0 or W <= 0:
                return None
            return (H, W)

        # Priority 1: any source skill's L3 obs body shape
        if src_skills_dir.is_dir():
            for src_path in src_skills_dir.glob("*.json"):
                if src_path.name.startswith("_"):
                    continue
                try:
                    with open(src_path) as f:
                        sk = json.load(f)
                except Exception:
                    continue
                obs_list = sk.get("l3", {}).get("observations", []) or []
                for o in obs_list:
                    sh = _shape_from_body(o.get("body"))
                    if sh:
                        return sh

        # Priority 2: state_snapshot elites
        src_root = src_skills_dir.parent
        gen_dirs = sorted(
            [p for p in src_root.glob("generation_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        )
        for gd in reversed(gen_dirs):
            snap = gd / "state_snapshot.json"
            if not snap.exists():
                continue
            try:
                data = json.loads(snap.read_text())
            except Exception:
                continue
            for e in (data.get("elites") or []):
                sh = _shape_from_body(e.get("body"))
                if sh:
                    return sh

        # Priority 3: designs.json
        for gd in gen_dirs:
            dj = gd / "designs.json"
            if not dj.exists():
                continue
            try:
                designs = json.loads(dj.read_text())
            except Exception:
                continue
            if isinstance(designs, list):
                for d in designs:
                    sh = _shape_from_body(d.get("body"))
                    if sh:
                        return sh

        # Fallback: current STRUCTURE_SHAPE (warn)
        cur = tuple(self.config.STRUCTURE_SHAPE)
        print(f"  [Import] WARNING: could not infer source_grid from {src_skills_dir}; "
              f"falling back to current STRUCTURE_SHAPE={cur}")
        return cur

    def _import_skill_library(
        self,
        src_skills_dir: Path,
        retrieve_as_task: str | None,
    ) -> tuple[list[str], list[str]]:
        """Copy source skill JSONs into self.skills_dir as frozen imported skills.

        Returns (imported_skill_ids, skipped_skill_ids).
        Strips l3 obs and obs_ids; tags status.origin='imported' + source_grid /
        source_exp / imported_at; optionally appends retrieve_as_task to task_family
        (modifying the COPY only, never the source on disk).
        """
        imported, skipped = [], []
        if not src_skills_dir.is_dir():
            raise FileNotFoundError(f"--init-skill-library path not found: {src_skills_dir}")

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        source_exp = src_skills_dir.parent.name
        imported_at = datetime.now(timezone.utc).isoformat()
        src_grid = list(self._infer_source_grid(src_skills_dir))

        for src_path in src_skills_dir.glob("*.json"):
            if src_path.name.startswith("_"):
                continue
            try:
                with open(src_path) as f:
                    skill = json.load(f)
            except Exception as e:
                print(f"  [Import] Failed to load {src_path.name}: {e}")
                skipped.append(src_path.stem)
                continue

            skill_id = skill.get("skill_id", src_path.stem)
            task_family = skill.get("task_family") or []
            if not task_family:
                skipped.append(skill_id)
                continue

            # Strip L3 obs and L2 leaf obs_ids
            skill["l3"] = {"observations": [], "next_obs_id_counter": 0}
            for bucket in ("positive", "negative"):
                for leaf in skill.get("l2", {}).get(bucket, []):
                    # Freeze source's true evidence count BEFORE stripping obs_ids
                    existing_birth = int(leaf.get("birth_obs_count", 0) or 0)
                    src_obs_count = len(leaf.get("obs_ids") or [])
                    leaf["birth_obs_count"] = max(existing_birth, src_obs_count)
                    leaf.pop("obs_ids", None)

            # Status: tag origin, source_grid, source_exp, imported_at
            status = skill.setdefault("status", {})
            status["origin"] = "imported"
            status["source_grid"] = src_grid
            status["source_exp"] = source_exp
            status["imported_at"] = imported_at
            status.pop("absorbed_skill_ids", None)
            skill.pop("absorbed_skill_ids", None)
            status["generation_born"] = -1
            status["generation_updated"] = -1

            # Cross-grid task_family extension (copy-only)
            if retrieve_as_task and retrieve_as_task not in task_family:
                task_family.append(retrieve_as_task)
                skill["task_family"] = task_family

            with open(self.skills_dir / src_path.name, "w") as f:
                json.dump(skill, f, indent=2)
            imported.append(skill_id)

        return imported, skipped

    def _import_elite_pool(
        self,
        src_skills_dir: Path,
        n_static_elites: int,
    ) -> list[dict]:
        """Extract top-K elite designs from source experiment.

        Source experiment ROOT is src_skills_dir.parent (state_snapshot.json
        and generation_*/ live there, not inside skills/).

        Priority:
          1. <root>/generation_<max>/state_snapshot.json -> elites field
          2. Fallback: scan all <root>/generation_*/designs.json by fitness desc.

        Writes <new_skills_dir>/_static_elites.json. Returns the elites list.
        """
        src_root = src_skills_dir.parent
        gen_dirs = sorted(
            [p for p in src_root.glob("generation_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        )
        elites: list[dict] = []

        # Priority 1: latest generation's state_snapshot.json
        for gd in reversed(gen_dirs):
            snap = gd / "state_snapshot.json"
            if not snap.exists():
                continue
            try:
                data = json.loads(snap.read_text())
                snap_elites = data.get("elites") or []
                if snap_elites:
                    elites = snap_elites[:n_static_elites]
                    break
            except Exception:
                continue

        # Fallback: scan designs.json files
        if not elites:
            all_designs: list[dict] = []
            for gd in gen_dirs:
                dj = gd / "designs.json"
                if not dj.exists():
                    continue
                try:
                    designs = json.loads(dj.read_text())
                    if isinstance(designs, list):
                        for d in designs:
                            if d.get("fitness") is not None:
                                all_designs.append(d)
                except Exception:
                    continue
            all_designs.sort(key=lambda d: d.get("fitness", float("-inf")), reverse=True)
            elites = []
            seen_bodies = set()
            for d in all_designs:
                body_key = json.dumps(d.get("body"))
                if body_key in seen_bodies:
                    continue
                seen_bodies.add(body_key)
                elites.append({
                    "label": f"E{len(elites)}",
                    "fitness": d.get("fitness"),
                    "body": d.get("body"),
                    "source_generation": d.get("generation"),
                })
                if len(elites) >= n_static_elites:
                    break

        payload = {
            "source_exp": src_root.name,
            "source_grid": list(self._infer_source_grid(src_skills_dir)),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "elites": elites,
        }
        with open(self.skills_dir / "_static_elites.json", "w") as f:
            json.dump(payload, f, indent=2)
        return elites

    def _import_static_payload(
        self,
        src_skills_dir: Path,
        retrieve_as_task: str | None,
        elite_inject: bool,
        n_static_elites: int,
        static_active_gens: int,
    ) -> int | None:
        """Orchestrate import: skill library + (T16) elite pool + provenance.

        Returns effective static_active_gens (None when no skill imported / cold-start fallback).
        """
        imported_ids, skipped_ids = self._import_skill_library(
            src_skills_dir=src_skills_dir,
            retrieve_as_task=retrieve_as_task,
        )

        effective_mode = "warm_start" if imported_ids else "cold_start_fallback"

        # Elite pool: only when warm AND elite_inject
        if effective_mode == "warm_start" and elite_inject:
            self.static_elites = self._import_elite_pool(
                src_skills_dir=src_skills_dir,
                n_static_elites=n_static_elites,
            )
        else:
            self.static_elites = []

        provenance = {
            "source_path": str(src_skills_dir.resolve()),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "imported_skill_ids": imported_ids,
            "skipped_skill_ids": skipped_ids,
            "retrieve_as_task": retrieve_as_task,
            "exp_name": self.exp_name,
            "env_name": self.env_name,
            "elite_inject_enabled": bool(elite_inject and effective_mode == "warm_start"),
            "n_static_elites": n_static_elites if (elite_inject and effective_mode == "warm_start") else 0,
            "effective_mode": effective_mode,
        }
        with open(self.skills_dir / "_origin_provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
        print(f"  [Import] {effective_mode}: {len(imported_ids)} skills imported, "
              f"{len(self.static_elites)} elites (skipped: {len(skipped_ids)})")

        if effective_mode == "cold_start_fallback":
            return None
        return static_active_gens

    def _load_init_population(self, init_path: Path) -> list[dict]:
        """Load a fixed Gen 0 population with pre-computed fitness.

        Expected directory layout (matches GA baseline output):
          init_path/
            structure/
              0.npz        (body = arr_0, connections = arr_1)
              1.npz
              ...
            output.txt     (lines of "<label> <fitness>")

        Returns a list of design dicts ready to be placed in self.elites /
        self._seen_designs at Gen 0 without invoking PPO.
        """
        if not init_path.is_dir():
            raise FileNotFoundError(f"--init-designs-path not found: {init_path}")
        structure_dir = init_path / "structure"
        if not structure_dir.is_dir():
            raise FileNotFoundError(f"Expected structure/ subdir under {init_path}")
        fitness_file = init_path / "output.txt"
        if not fitness_file.exists():
            raise FileNotFoundError(f"Expected output.txt under {init_path}")

        fitness_map = {}
        with open(fitness_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        fitness_map[int(parts[0])] = float(parts[1])
                    except ValueError:
                        continue

        init_designs = []
        for npz_file in sorted(structure_dir.glob("*.npz")):
            try:
                label = int(npz_file.stem)
            except ValueError:
                continue
            if label not in fitness_map:
                print(f"  [Init] Skipping {npz_file.name}: no fitness entry in output.txt")
                continue
            data = np.load(npz_file)
            init_designs.append({
                "body": data["arr_0"],
                "connections": data["arr_1"],
                "label": self._next_label(),
                "fitness": fitness_map[label],
                "reasoning": "",
                "based_on_skill": None,
                "_proposal_path": "I",
                "eval_status": "ok",
                "generation": 0,
            })
        if not init_designs:
            raise RuntimeError(f"No valid init designs loaded from {init_path}")
        return init_designs

    def run(self):
        summary = []
        for generation in range(self.config.TOTAL_GENERATIONS):
            print(f"\n=== Generation {generation} ===")

            ops_log_path = str(self.logs_dir / "skill_ops.json")

            # Gen 0 with fixed init population: bootstrap mode.
            # Init designs are evidence-free seeds — they only inspire skill creation
            # via NEW_INSIGHT (v2). They do NOT enter any skill.L3, do NOT update any
            # leaf/status statistics, and do NOT enter unassigned_pool. All real stats
            # accumulate from gen 1+ real mutations.
            if generation == 0 and self._init_designs is not None:
                results = []
                for d in self._init_designs:
                    r = dict(d)
                    r["generation"] = 0
                    results.append(r)
                    self._seen_designs[hashable(r["body"])] = r["fitness"]
                self._update_elites(results)
                print(f"  [Gen 0] Loaded init population: {len(results)} designs, "
                      f"best={max(r['fitness'] for r in results):.3f}")

                # Build pseudo-obs for v2 run_new_insight's fallback path.
                # Synthetic baseline: init designs have no real parent. Use cohort median
                # as virtual parent_fitness so gain = fitness - median acts as a
                # "percentile rank within init batch" sortable signal for NEW_INSIGHT
                # filtering. These pseudo-obs are NOT pushed to pool, so v2's migration
                # logic (which only migrates pool obs) leaves new_skill.L3 empty.
                median_fit = statistics.median(r["fitness"] for r in results)
                pseudo_obs = [
                    {
                        "obs_id": i,  # local index — init designs do not get global obs_id
                        "body": r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"],
                        "fitness": float(r["fitness"]),
                        "parent_body_hash": None,
                        "parent_fitness": float(median_fit),
                        "gain": float(r["fitness"] - median_fit),
                        "gen_observed": 0,
                        "source": "path_g",
                    }
                    for i, r in enumerate(results)
                ]

                # Call v2 NEW_INSIGHT. Pool is empty -> takes fallback path on pseudo_obs.
                # Pseudo-obs are not in pool -> inspired_obs_ids migration loop is no-op.
                # Result: new skill is created with empty L2/L3 and zeroed status.
                new_skill = run_new_insight(self.skill_library, 0, pseudo_obs, self.env_name)
                if new_skill is not None:
                    print(f"  [Init NewInsight] Created bootstrap skill '{new_skill['skill_id']}' "
                          f"(empty L2/L3 — gen 1+ will populate from real evidence)")

                # No run_diagnosis call: skill.L3 is empty, no obs has gen_observed=0
                # in any skill, so v2 run_diagnosis would be a no-op anyway.

                self._update_parent_history_index(results)
                gen_summary = self._log_generation(generation, results)
                summary.append(gen_summary)
                dump_json(summary, self.logs_dir / "summary.json")
                continue

            # Step 1: Consolidation Scan (gen >= 1, only if library changed, min 4 skills)
            if (
                not self.config.ABLATE_MERGE
                and generation > 0
                and self.skill_library.has_pending_changes()
            ):
                active_skills = self.skill_library.retrieve(self.env_name)
                if len(active_skills) >= 4:
                    self.skill_library.consolidate(
                        self.env_name, generation,
                        ops_log_path=ops_log_path,
                    )
                else:
                    print(f"  [Consolidation] Skipped — only {len(active_skills)} active skills (min 4)")
                self.skill_library.clear_change_flag()

            # Step 1b: Retrieve relevant skills
            skills = self._step1_retrieve(generation)

            # Step 2: Propose designs via LLM
            designs = self._step2_propose(skills, generation)

            # Budget control: truncate designs if we'd exceed MAX_EVALUATIONS
            max_evals = self.config.MAX_EVALUATIONS
            if max_evals > 0 and designs:
                remaining = max_evals - self._eval_count
                if remaining <= 0:
                    print(f"  [Generation {generation}] Evaluation budget exhausted "
                          f"({self._eval_count}/{max_evals}) — stopping.")
                    break
                if len(designs) > remaining:
                    print(f"  [Generation {generation}] Truncating {len(designs)} → {remaining} "
                          f"designs (budget: {self._eval_count + remaining}/{max_evals})")
                    designs = designs[:remaining]

            # Skip generation if no valid designs (e.g. API completely down)
            if not designs:
                print(f"  [Generation {generation}] No valid designs — skipping.")
                gen_summary = {
                    "generation": generation,
                    "n_designs": 0,
                    "best_fitness": None,
                    "mean_fitness": None,
                    "skills_total": len(self.skill_library.retrieve(self.env_name)),
                    "elites_best": self.elites[0]["fitness"] if self.elites else None,
                    "skipped": True,
                }
                summary.append(gen_summary)
                dump_json(summary, self.logs_dir / "summary.json")
                continue

            # Evaluate designs with parallel PPO
            results = self._step3_evaluate(designs, generation)
            self._eval_count += len(results)
            for r in results:
                r["generation"] = generation

            # Update global elites
            self._update_elites(results)

            # v2 pipeline: pre-filter → attribution → L1 new insight → L2 diagnosis
            # evaluated_designs must be a list of dicts with keys:
            #   body, fitness, parent_body_hash, parent_fitness, gain, eval_status,
            #   based_on_skill (Path A only), intended_leaf_id (Path A only)
            evaluated_designs = []
            for r in results:
                parent_fitness = r.get("_parent_fitness")
                fitness = r.get("fitness")
                gain = (
                    float(fitness - parent_fitness)
                    if (fitness is not None and parent_fitness is not None)
                    else None
                )
                evaluated_designs.append({
                    "body": r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"],
                    "fitness": float(fitness) if fitness is not None else None,
                    "parent_body_hash": r.get("_parent_body_hash"),
                    "parent_fitness": (
                        float(parent_fitness) if parent_fitness is not None else None
                    ),
                    "gain": gain,
                    "eval_status": r.get("eval_status", "ok"),
                    "based_on_skill": r.get("based_on_skill"),
                    "intended_leaf_id": r.get("intended_leaf_id"),
                })
            summary_v2 = run_one_generation(
                library=self.skill_library,
                raw_obs=evaluated_designs,
                current_gen=generation,
                task_name=self.env_name,
                ablate_llm_diagnose=self.config.ABLATE_DIAGNOSE,
            )
            print(
                f"  [v2 pipeline] gen={generation} kept={summary_v2['kept_count']} "
                f"dropped={summary_v2['dropped_count']} new_skill={summary_v2['new_skill']} "
                f"diagnosed_skills={summary_v2['diagnosed_skill_count']} "
                f"pool_size={summary_v2['pool_size']}"
            )

            # Consolidate stays unchanged (orthogonal stage)
            if not self.config.ABLATE_MERGE and self.skill_library.has_pending_changes():
                self.skill_library.consolidate(
                    self.env_name, generation, ops_log_path=ops_log_path,
                )
                self.skill_library.clear_change_flag()

            # Log generation
            gen_summary = self._log_generation(generation, results)
            gen_summary["v2_pipeline"] = summary_v2
            self._update_parent_history_index(results)
            summary.append(gen_summary)

            # Write rolling summary
            dump_json(summary, self.logs_dir / "summary.json")

            print(
                f"Generation {generation}: "
                f"best={gen_summary['best_fitness']:.3f}, "
                f"mean={gen_summary['mean_fitness']:.3f}, "
                f"skills_total={gen_summary['skills_total']}"
            )

        return summary

    # -------------------------------------------------------------------------
    # Step 1 — Retrieve
    # -------------------------------------------------------------------------
    def _step1_retrieve(self, generation: int) -> list[dict]:
        skills = self.skill_library.retrieve(self.env_name, current_gen=generation)
        print(f"  Retrieved {len(skills)} skills from library (gen={generation})")
        return skills

    # -------------------------------------------------------------------------
    # Step 2 — Propose
    # -------------------------------------------------------------------------
    def _step2_propose(self, skills: list[dict], generation: int) -> list[dict]:
        """Propose designs via two parallel paths:
          Path A (LLM): skill-guided elite mutation with local retry only
          Path G (GA):  stochastic voxel-level mutation for global exploration

        Overflow rule: if LLM can't fill its slots, extras go to GA.
        """
        target = self.config.N_DESIGNS_PER_GEN
        n_llm_target = self.config.N_LLM_SLOTS
        n_ga_target = self.config.N_GA_SLOTS

        # --- Gen 0: no elites yet, all LLM init (GA needs parents) ---
        if generation == 0 or not self.elites:
            print(f"  Gen 0 init: requesting {target} diverse designs (no elites for GA)")
            return self._propose_llm_init(target)

        # --- Path A: one pass only, with per-parent local retry ---
        raw_llm_designs = self._propose_path_a(skills, generation, n_llm_target, round_idx=0)
        llm_designs, seen_this_batch = self._dedup_new_designs(raw_llm_designs)

        # --- Ablation: pure-LLM disables Path G entirely (no overflow, no GA) ---
        if self.config.ABLATE_GA:
            print(f"  [Ablation] Pure LLM mode: skipping Path G, no overflow")
            print(f"  Slot allocation: A(LLM)={len(llm_designs)}/{n_llm_target}, G(GA)=0 (ablated)")
            return llm_designs

        # --- Overflow: unfilled LLM slots go to GA ---
        n_overflow = n_llm_target - len(llm_designs)
        actual_ga_target = n_ga_target + n_overflow
        if n_overflow > 0:
            print(f"  [Overflow] LLM filled {len(llm_designs)}/{n_llm_target} → "
                  f"{n_overflow} overflow slots to GA (total GA: {actual_ga_target})")

        print(f"  Slot allocation: A(LLM)={len(llm_designs)}, G(GA)={actual_ga_target}")

        # --- Path G: GA random mutation with inline dedup (same pattern as GA baseline) ---
        parents = self.elites[: self.config.N_ELITE_PARENTS]
        all_designs = list(llm_designs)
        ga_accepted = 0
        max_ga_attempts = actual_ga_target * 50  # match GA baseline's generous retry budget

        print(f"  [Path G] GA mutation: requesting {actual_ga_target} designs from {len(parents)} elite parents")
        while ga_accepted < actual_ga_target and max_ga_attempts > 0:
            parent = random.choice(parents)
            parent_body = np.array(parent["body"], dtype=int)
            result = mutate(parent_body.copy(), mutation_rate=0.1, num_attempts=50)
            max_ga_attempts -= 1

            if result is None:
                continue

            body, connections = result
            h = hashable(body)
            if h in self._seen_designs or h in seen_this_batch:
                continue

            label = self._next_label()
            design = {
                "body": body,
                "connections": connections,
                "label": label,
                "reasoning": "",
                "based_on_skill": None,
                "_proposal_path": "G",
            }
            self._attach_parent_context(design, parent)
            all_designs.append(design)
            seen_this_batch.add(h)
            ga_accepted += 1

        print(f"  [Path G] Produced {ga_accepted}/{actual_ga_target} unique designs")

        if len(all_designs) < target:
            print(f"  Warning: only {len(all_designs)}/{target} unique designs total")
        return all_designs

    def _propose_llm_init(self, n_designs: int) -> list[dict]:
        """Gen 0 proposal: diverse LLM-generated designs (no elites available)."""
        prompt = PROPOSE_INIT_PROMPT.format(
            task_desc=TASK_DESCRIPTIONS[self.env_name],
            voxel_legend=VOXEL_LEGEND,
            n_designs=n_designs,
        )
        batch = self._propose_init_with_retry(prompt, n_designs)
        for d in batch:
            d["_proposal_path"] = "A"

        # Dedup against seen
        return self._dedup_new_designs(batch)[0]

    def _dedup_new_designs(self, designs: list[dict]) -> tuple[list[dict], set[str]]:
        unique = []
        seen_this_batch: set[str] = set()
        for d in designs:
            h = hashable(d["body"])
            if h not in self._seen_designs and h not in seen_this_batch:
                unique.append(d)
                seen_this_batch.add(h)
            else:
                print(f"    Skipping duplicate design (label={d['label']})")
        if len(unique) < len(designs):
            print(f"  Dedup: {len(designs)} proposed -> {len(unique)} new, {len(designs) - len(unique)} duplicates")
        return unique, seen_this_batch

    def _propose_path_a(
        self, skills: list[dict], generation: int, n_designs: int, round_idx: int
    ) -> list[dict]:
        """Path A: per-slot skill sampling with per-parent batching."""
        valid_designs = []
        parents = self.elites[: self.config.N_ELITE_PARENTS]
        skill_weights, weight_debug = self._compute_skill_sampling_weights(skills)
        # Distribute n_designs across available parents
        n_per_parent = n_designs // len(parents)
        n_extra = n_designs % len(parents)
        if round_idx == 0:
            print(f"  [Path A] Mutating {len(parents)} elite parents, ~{n_per_parent} children each")
            if weight_debug:
                ranked = sorted(weight_debug, key=lambda row: row["weight"], reverse=True)
                preview = ", ".join(
                    f"{row['skill_id']}={row['weight']:.3f}"
                    for row in ranked[:5]
                )
                print(f"  [Path A] Skill weights: {preview}")

        for i, parent in enumerate(parents):
            n = n_per_parent + (1 if i < n_extra else 0)
            if n <= 0:
                continue
            slot_assignments = self._sample_skill_assignments(skills, skill_weights, n)
            batch = self._propose_parent_batch(
                parent=parent,
                slot_assignments=slot_assignments,
                mutation_range=self.config.MUTATION_RANGE,
                generation=generation,
            )
            for d in batch:
                d["_proposal_path"] = "A"
                self._attach_parent_context(d, parent)
            valid_designs.extend(batch)

        return valid_designs

    def _compute_skill_sampling_weights(self, skills: list[dict]) -> tuple[list[float], list[dict]]:
        """Compute Laplace-smoothed skill weights from clipped positive fitness gains."""
        if not skills:
            return [], []

        alias_map = self._build_skill_alias_map(skills)
        gain_by_skill: dict[str, list[float]] = {}
        for entries in self._parent_history_index.values():
            for entry in entries:
                if entry.get("proposal_path") != "A":
                    continue
                skill_id = entry.get("based_on_skill")
                if not skill_id:
                    continue
                skill_id = alias_map.get(skill_id, skill_id)
                parent_fitness = entry.get("parent_fitness")
                child_fitness = entry.get("child_fitness")
                if parent_fitness is None or child_fitness is None:
                    continue
                diff = float(child_fitness - parent_fitness)
                gain = min(1.0, max(0.0, diff / SKILL_WEIGHT_DELTA_CAP))
                gain_by_skill.setdefault(skill_id, []).append(gain)

        weights = []
        debug_rows = []
        for skill in skills:
            skill_id = skill.get("skill_id", "unknown")
            gains = gain_by_skill.get(skill_id, [])
            n = len(gains)
            weight = (sum(gains) + 1.0) / (n + 2.0)
            weights.append(weight)
            debug_rows.append({
                "skill_id": skill_id,
                "samples": n,
                "mean_gain": float(np.mean(gains)) if gains else 0.0,
                "weight": weight,
            })

        return weights, debug_rows

    def _build_skill_alias_map(self, skills: list[dict]) -> dict[str, str]:
        alias_map = {}
        for skill in skills:
            canonical_id = skill.get("skill_id")
            if not canonical_id:
                continue
            alias_map[canonical_id] = canonical_id
            status = skill.get("status", {})
            if isinstance(status, dict):
                absorbed_ids = status.get("absorbed_skill_ids", [])
                if isinstance(absorbed_ids, list):
                    for absorbed_id in absorbed_ids:
                        if absorbed_id:
                            alias_map[absorbed_id] = canonical_id
        return alias_map

    def _sample_skill_assignments(
        self,
        skills: list[dict],
        skill_weights: list[float],
        n_slots: int,
    ) -> list[dict]:
        assignments = []
        if skills:
            probs = np.array(skill_weights, dtype=float)
            probs = probs / probs.sum()
            sampled_indices = np.random.choice(len(skills), size=n_slots, p=probs)
            for slot_index, sampled_idx in enumerate(sampled_indices):
                sampled_skill = skills[int(sampled_idx)]
                assignments.append({
                    "slot_index": slot_index,
                    "skill": sampled_skill,
                    "skill_id": sampled_skill.get("skill_id"),
                })
        else:
            for slot_index in range(n_slots):
                assignments.append({
                    "slot_index": slot_index,
                    "skill": None,
                    "skill_id": None,
                })
        return assignments

    def _propose_parent_batch(
        self,
        parent: dict,
        slot_assignments: list[dict],
        mutation_range: str,
        generation: int,
        max_retry_rounds: int = 5,
    ) -> list[dict]:
        if not slot_assignments:
            return []

        valid_by_slot = {}
        remaining_assignments = list(slot_assignments)
        retry_note = ""

        # Compute static prior phase status + transfer/reference blocks ONCE per call
        K = self._effective_static_active_gens
        static_active = (K is not None) and (generation < K)
        src_grid = self._source_grid_tuple()
        cur_grid = tuple(self.config.STRUCTURE_SHAPE)
        same_grid = src_grid == cur_grid
        source_exp = self._source_exp_name() or ""

        transfer_context_block = _format_transfer_context_block(
            static_active=static_active,
            same_grid=same_grid,
            elite_inject=self._elite_inject_enabled,
            source_grid=src_grid,
            current_grid=cur_grid,
            source_exp=source_exp,
            current_env=self.env_name,
        )
        static_reference_block = _format_static_reference_block(
            static_active=static_active,
            elite_inject=self._elite_inject_enabled,
            elites=self.static_elites,
        )

        for retry_round in range(max_retry_rounds + 1):
            if not remaining_assignments:
                break

            parent_body_str = _format_body(parent["body"])
            history_block = self._format_parent_history_block(parent)
            prompt = PROPOSE_MUTATE_PROMPT.format(
                task_desc=TASK_DESCRIPTIONS[self.env_name],
                voxel_legend=VOXEL_LEGEND,
                parent_fitness=parent["fitness"],
                parent_body=parent_body_str,
                skill_assignments_block=_format_skill_assignments_block(
                    remaining_assignments,
                    l1_only=self.config.ABLATE_L2_INJECTION,
                ),
                history_block=history_block,
                n_designs=len(remaining_assignments),
                mutation_range=mutation_range,
                transfer_context_block=transfer_context_block,
                static_reference_block=static_reference_block,
            )
            full_prompt = prompt
            if retry_note:
                full_prompt += f"\n\nNote from previous attempt: {retry_note}"

            messages = [{"role": "user", "content": full_prompt}]
            try:
                response = call_llm(messages)
                raw_designs = response.get("designs", [])
            except Exception as e:
                print(f"  [Propose] LLM call failed: {e}")
                raw_designs = []

            expected_by_slot = {
                assignment["slot_index"]: assignment for assignment in remaining_assignments
            }
            invalid_feedback = []
            newly_valid = {}

            for item in raw_designs:
                slot_index = item.get("slot_index")
                if not isinstance(slot_index, int) or slot_index not in expected_by_slot:
                    invalid_feedback.append((item.get("body"), f"slot_index must be one of {sorted(expected_by_slot)}"))
                    continue
                if slot_index in newly_valid:
                    invalid_feedback.append((item.get("body"), f"Duplicate slot_index {slot_index}"))
                    continue

                body_list = item.get("body")
                if body_list is None:
                    invalid_feedback.append((None, f"Missing body for slot_index {slot_index}"))
                    continue
                try:
                    body = np.array(body_list, dtype=int)
                except Exception:
                    invalid_feedback.append((None, f"Could not parse body as integer array for slot_index {slot_index}"))
                    continue

                if body.shape != tuple(self.config.STRUCTURE_SHAPE):
                    invalid_feedback.append((
                        body_list,
                        f"Wrong shape {body.shape}, expected {self.config.STRUCTURE_SHAPE} for slot_index {slot_index}",
                    ))
                    continue

                if not is_connected(body):
                    invalid_feedback.append((body_list, f"Not fully connected for slot_index {slot_index}"))
                    continue

                if not has_actuator(body):
                    invalid_feedback.append((body_list, f"No actuator voxels for slot_index {slot_index}"))
                    continue

                expected_skill_id = expected_by_slot[slot_index]["skill_id"]
                raw_skill = item.get("based_on_skill") or None
                if raw_skill != expected_skill_id:
                    invalid_feedback.append((
                        body_list,
                        f"based_on_skill must be {expected_skill_id!r} for slot_index {slot_index}, not {raw_skill!r}",
                    ))
                    continue

                newly_valid[slot_index] = {
                    "body": body,
                    "connections": get_full_connectivity(body),
                    "label": self._next_label(),
                    "reasoning": item.get("reasoning", ""),
                    "based_on_skill": expected_skill_id,
                    "intended_leaf_id": item.get("intended_leaf_id"),
                }

            valid_by_slot.update(newly_valid)

            remaining_assignments = [
                assignment for assignment in remaining_assignments
                if assignment["slot_index"] not in newly_valid
            ]
            if not remaining_assignments:
                break

            if retry_round >= max_retry_rounds:
                print(
                    f"  [Propose] Parent={parent.get('label')} stopped with "
                    f"{len(remaining_assignments)} slots still missing after {max_retry_rounds} retries"
                )
                break

            note_lines = [
                f"Previous attempts produced valid designs for {len(valid_by_slot)} of {len(slot_assignments)} total slots.",
                f"Please regenerate ONLY these missing slot_index values: {[a['slot_index'] for a in remaining_assignments]}",
                "The following designs were rejected (showing up to 3):",
            ]
            for body_list, reason in invalid_feedback[:3]:
                if body_list is not None:
                    note_lines.append(f"  - body={body_list} -> {reason}")
                else:
                    note_lines.append(f"  - {reason}")
            note_lines.append(
                "Return exactly one design for each missing slot_index, and keep all slot_index / based_on_skill assignments exact."
            )
            retry_note = "\n".join(note_lines)
            print(
                f"  [Propose] Retrying missing slots for parent={parent.get('label')}: "
                f"{len(remaining_assignments)} still missing (retry {retry_round + 1}/{max_retry_rounds})"
            )

        return [
            valid_by_slot[assignment["slot_index"]]
            for assignment in slot_assignments
            if assignment["slot_index"] in valid_by_slot
        ]

    def _propose_init_with_retry(
        self,
        prompt: str,
        n_designs: int,
        retry_note: str = "",
    ) -> list[dict]:
        full_prompt = prompt
        if retry_note:
            full_prompt += f"\n\nNote from previous attempt: {retry_note}"

        messages = [{"role": "user", "content": full_prompt}]
        try:
            response = call_llm(messages)
            raw_designs = response.get("designs", [])
        except Exception as e:
            print(f"  [Propose] LLM call failed: {e}")
            raw_designs = []

        valid = []
        invalid_feedback = []  # list of (body_list_or_None, reason_str)
        for item in raw_designs:
            body_list = item.get("body")
            if body_list is None:
                continue
            try:
                body = np.array(body_list, dtype=int)
            except Exception:
                invalid_feedback.append((None, "Could not parse body as integer array"))
                continue

            if body.shape != tuple(self.config.STRUCTURE_SHAPE):
                invalid_feedback.append((
                    body_list,
                    f"Wrong shape {body.shape}, expected {self.config.STRUCTURE_SHAPE}",
                ))
                continue

            if not is_connected(body):
                invalid_feedback.append((body_list, "Not fully connected (isolated voxel groups)"))
                continue

            if not has_actuator(body):
                invalid_feedback.append((body_list, "No actuator voxels (need type 3=H_ACT or 4=V_ACT)"))
                continue

            connections = get_full_connectivity(body)
            label = self._next_label()
            valid.append({
                "body": body,
                "connections": connections,
                "label": label,
                "reasoning": item.get("reasoning", ""),
                "based_on_skill": None,
            })

        # If too few valid designs, retry once with structured error feedback
        min_required = max(1, n_designs // 2)
        if len(valid) < min_required and not retry_note:
            note_lines = [
                f"Previous attempt produced only {len(valid)} valid designs out of {n_designs} requested.",
                "The following designs were rejected (showing up to 3):",
            ]
            for body_list, reason in invalid_feedback[:3]:
                if body_list is not None:
                    note_lines.append(f"  - body={body_list} → {reason}")
                else:
                    note_lines.append(f"  - {reason}")
            note_lines.append(
                "Please fix these issues: every design must be fully connected "
                "(all non-empty voxels reachable via 4-connectivity) and contain "
                "at least one actuator (3=H_ACT or 4=V_ACT)."
            )
            note = "\n".join(note_lines)
            print(f"  [Propose] Retrying — only {len(valid)} valid designs, "
                  f"{len(invalid_feedback)} rejected")
            retry_valid = self._propose_init_with_retry(
                prompt,
                n_designs,
                retry_note=note,
            )
            seen = {d["label"] for d in valid}
            for d in retry_valid:
                if d["label"] not in seen:
                    valid.append(d)
                    seen.add(d["label"])

        return valid

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    def _step3_evaluate(self, designs: list[dict], generation: int) -> list[dict]:
        if not designs:
            print("  No designs to evaluate")
            return []

        gen_save_dir = str(self.save_root / "logs" / self.exp_name)
        results = evaluate_batch(
            designs=designs,
            env_name=self.env_name,
            save_dir=gen_save_dir,
            generation=generation,
            num_cores=self.config.NUM_CORES,
            ppo_args=self.config.PPO_ARGS,
        )

        # Post-process: mark suspicious (likely PPO crash) and register in seen cache
        for r in results:
            h = hashable(r["body"])
            if r["fitness"] == 0.0 and has_actuator(r["body"]):
                r["eval_status"] = "suspicious"
            else:
                r["eval_status"] = "ok"
                self._seen_designs[h] = r["fitness"]

        fitnesses = [r["fitness"] for r in results]
        n_sus = sum(1 for r in results if r["eval_status"] == "suspicious")
        status = f"  Evaluated {len(results)} designs: best={max(fitnesses):.3f}, mean={sum(fitnesses)/len(fitnesses):.3f}"
        if n_sus:
            status += f" ({n_sus} suspicious)"
        print(status)
        return results

    # -------------------------------------------------------------------------
    # Update elites
    # -------------------------------------------------------------------------
    def _update_elites(self, results: list[dict]):
        for r in results:
            self.elites.append({
                "label": r.get("label"),
                "body": r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"],
                "fitness": r["fitness"],
                "generation": r.get("generation", -1),
                "reasoning": r.get("reasoning", ""),
                "based_on_skill": r.get("based_on_skill"),
            })
        self.elites.sort(key=lambda e: e["fitness"], reverse=True)
        self.elites = self.elites[: self.config.ELITE_SIZE]

    # -------------------------------------------------------------------------
    # Step 2 (post-eval) — Evidence Attribution
    # -------------------------------------------------------------------------
    def _step2_evidence_attribution(
        self, results: list[dict], retrieved_skills: list[dict]
    ) -> tuple[dict[str, dict], float]:
        """Tier-based per-design attribution (v3).

        Only Path A (LLM skill-guided mutation) designs participate. For each
        Path A design, compute a tier tag:
            - positive:  fitness > gen_mean   (loose gate)
            - negative:  fitness < p25         (strict gate, bottom quartile)
            - middle:    otherwise             (dropped, not fed to DIAGNOSE)

        Then bucket designs by their `based_on_skill`. Each bucket is either
        a positives list, a negatives list, or both — middle-tier designs are
        excluded even if they reference a skill.

        Hard guard: if Path A has fewer than 2 designs, skip DIAGNOSE entirely
        for the whole generation.

        Returns: (buckets, gen_mean)
            buckets: {skill_id: {"positives": [...], "negatives": [...]}}
            gen_mean: mean fitness of Path A designs (for logging)
        """
        if not results:
            return {}, 0.0

        retrieved_ids = {s.get("skill_id") for s in retrieved_skills if s.get("skill_id")}

        path_a_results = [
            r for r in results
            if r.get("_proposal_path") == "A" and r.get("eval_status") != "suspicious"
        ]
        if len(path_a_results) < 2:
            print(f"  [Attribution] Path A has {len(path_a_results)} clean designs (<2), "
                  f"skipping DIAGNOSE for the whole generation")
            return {}, 0.0

        path_a_fitnesses = [r["fitness"] for r in path_a_results]
        gen_mean = sum(path_a_fitnesses) / len(path_a_fitnesses)
        sorted_fits = sorted(path_a_fitnesses)
        p25_index = len(sorted_fits) // 4
        p25 = sorted_fits[p25_index]
        print(f"  [Attribution] Path A: n={len(path_a_results)}, "
              f"gen_mean={gen_mean:.3f}, p25={p25:.3f}")

        # Assign a tier to every Path A design
        buckets: dict[str, dict[str, list]] = {}
        n_pos = n_neg = n_mid = 0
        for r in path_a_results:
            fitness = r["fitness"]
            if fitness > gen_mean:
                tier = "positive"
                n_pos += 1
            elif fitness < p25:
                tier = "negative"
                n_neg += 1
            else:
                tier = "middle"
                n_mid += 1
                continue  # middle-tier designs do not feed DIAGNOSE

            skill_id = r.get("based_on_skill")
            if not skill_id or skill_id not in retrieved_ids:
                continue

            bucket = buckets.setdefault(skill_id, {"positives": [], "negatives": []})
            entry = {
                "body": r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"],
                "fitness": fitness,
                "parent_body_hash": r.get("_parent_body_hash"),
                "parent_fitness": r.get("_parent_fitness"),
                "gain": (
                    float(fitness - r["_parent_fitness"])
                    if r.get("_parent_fitness") is not None else None
                ),
                "label": r.get("label"),
            }
            if tier == "positive":
                bucket["positives"].append(entry)
            else:
                bucket["negatives"].append(entry)

        print(f"  [Attribution] tiers: positive={n_pos}, negative={n_neg}, "
              f"middle={n_mid} (dropped)")
        for skill_id, b in buckets.items():
            print(f"  [Attribution] '{skill_id}': "
                  f"positives={len(b['positives'])}, negatives={len(b['negatives'])}")

        return buckets, gen_mean

    # -------------------------------------------------------------------------
    # Step 3 (post-eval) — Diagnosis
    # -------------------------------------------------------------------------
    def _step3_diagnosis(
        self, buckets: dict[str, dict], gen_mean: float,
        results: list[dict],
        retrieved_skills: list[dict], generation: int, ops_log_path: str
    ):
        """v3 append-only diagnosis.

        For each skill with a non-empty positives or negatives bucket, call
        SkillLibrary.diagnose() which appends L3 observations and updates L2
        leaves. Skill utility is handled upstream by biased sampling rather
        than any hard-retire rule.

        Unretrieved diagnosis is kept: if top designs match a non-retrieved
        skill's pattern, expand that skill's task_family to include the
        current task.
        """
        # --- v3 append-only diagnose per attributed skill ---
        for skill_id, bucket in buckets.items():
            positives = bucket.get("positives", [])
            negatives = bucket.get("negatives", [])
            if not positives and not negatives:
                continue

            skill = self.skill_library.diagnose(
                skill_id, positives, negatives, generation,
            )
            if skill is None:
                continue

            self.skill_library._save(skill)
            self.skill_library.mark_changed()
            if ops_log_path:
                self.skill_library._append_ops_log(ops_log_path, [{
                    "generation": generation,
                    "action": "diagnose_append",
                    "skill_id": skill_id,
                    "n_positives": len(positives),
                    "n_negatives": len(negatives),
                    "trigger": "diagnose",
                }])

        # --- Unretrieved diagnosis: check if top designs match non-retrieved skills ---
        retrieved_ids = {s.get("skill_id") for s in retrieved_skills if s.get("skill_id")}
        all_skills = self.skill_library.retrieve_all()
        non_retrieved = [
            s for s in all_skills
            if s.get("skill_id") not in retrieved_ids
        ]
        if non_retrieved:
            top_results = sorted(results, key=lambda r: r["fitness"], reverse=True)[:3]
            if top_results:
                top_designs_text = _format_generation_results_no_skill(top_results)
                non_retrieved_text = "\n".join(
                    f"[{s.get('skill_id')}] task_family={s.get('task_family', [])} | "
                    f"L1 condition: {_skill_condition(s) or 'N/A'} | "
                    f"L2 positive rules: {_format_l2_rules_for_llm(_skill_positive_rules(s), 'positive')}"
                    for s in non_retrieved
                )
                prompt = UNRETRIEVED_DIAGNOSE_PROMPT.format(
                    current_task=self.env_name,
                    top_designs=top_designs_text,
                    non_retrieved_skills=non_retrieved_text,
                )
                try:
                    response = call_llm([{"role": "user", "content": prompt}])
                    matches = response.get("matches", [])
                    for m in matches:
                        sid = m.get("skill_id")
                        if sid:
                            print(f"  [Diagnosis-Unretrieved] Expanding '{sid}' to include "
                                  f"'{self.env_name}': {m.get('reason', '')}")
                            self.skill_library.expand_task_family(
                                sid, self.env_name, generation, ops_log_path=ops_log_path
                            )
                except Exception as e:
                    print(f"  [Diagnosis-Unretrieved] LLM call failed: {e}")

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log_generation(self, generation: int, results: list[dict]) -> dict:
        gen_dir = self.logs_dir / f"generation_{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Serialise designs
        designs_log = []
        for r in results:
            body = r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"]
            designs_log.append({
                "label": r["label"],
                "body": body,
                "generation": generation,
                "fitness": r["fitness"],
                "eval_status": r.get("eval_status"),
                "reasoning": r.get("reasoning", ""),
                "based_on_skill": r.get("based_on_skill"),
                "proposal_path": r.get("_proposal_path"),
                "parent_label": r.get("_parent_label"),
                "parent_body_hash": r.get("_parent_body_hash"),
            })

        dump_json(designs_log, gen_dir / "designs.json")

        fitnesses = [r["fitness"] for r in results]
        skills_all = self.skill_library.retrieve(self.env_name)

        # Telemetry: imported vs native breakdown for this generation.
        # Use plain retrieve (no phase filter) to count what's physically in the library.
        all_skills_for_task = self.skill_library.retrieve(self.env_name)
        imported_visible = [s for s in all_skills_for_task if _is_frozen(s)]
        native_visible = [s for s in all_skills_for_task if not _is_frozen(s)]

        # Designs that named an imported vs native skill in based_on_skill
        imported_ids = {s["skill_id"] for s in imported_visible}
        native_ids = {s["skill_id"] for s in native_visible}
        n_designs_using_imported = sum(1 for r in results if r.get("based_on_skill") in imported_ids)
        n_designs_using_native = sum(1 for r in results if r.get("based_on_skill") in native_ids)

        # Per-path breakdown
        path_stats = {}
        for path in ("A", "G"):
            path_fits = [r["fitness"] for r in results if r.get("_proposal_path") == path]
            if path_fits:
                path_stats[path] = {
                    "n": len(path_fits),
                    "best": max(path_fits),
                    "mean": sum(path_fits) / len(path_fits),
                }

        summary = {
            "generation": generation,
            "n_designs": len(results),
            "best_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "skills_total": len(skills_all),
            "elites_best": self.elites[0]["fitness"] if self.elites else 0.0,
            "skipped": False,
            "path_stats": path_stats,
            "n_imported_visible": len(imported_visible),
            "n_native_visible": len(native_visible),
            "n_designs_using_imported": n_designs_using_imported,
            "n_designs_using_native": n_designs_using_native,
            "ablations": {
                "diagnose": getattr(self.config, "ABLATE_DIAGNOSE", False),
                "merge": getattr(self.config, "ABLATE_MERGE", False),
                "pure_llm": getattr(self.config, "ABLATE_GA", False),
                "l2_injection": getattr(self.config, "ABLATE_L2_INJECTION", False),
            },
        }

        # Save FSM state snapshot: full skill library + elites at this generation
        all_skills = self.skill_library.retrieve_all(include_fully_retired=True)
        state_snapshot = {
            "generation": generation,
            "skills": all_skills,
            "elites": self.elites,
        }
        dump_json(state_snapshot, gen_dir / "state_snapshot.json")

        return summary

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _next_label(self) -> int:
        label = self._label_counter
        self._label_counter += 1
        return label

    def _attach_parent_context(self, design: dict, parent: dict):
        parent_body = _body_to_list(parent["body"])
        design["_parent_label"] = parent.get("label")
        design["_parent_fitness"] = parent.get("fitness")
        design["_parent_generation"] = parent.get("generation")
        design["_parent_body"] = parent_body
        design["_parent_body_hash"] = hashable(np.array(parent_body, dtype=int))

    def _source_grid_tuple(self) -> tuple[int, int]:
        """Return source_grid from any imported skill, else current grid.

        All imported skills share the same source_grid (set at import time),
        so any one of them is authoritative.
        """
        for skill in self.skill_library.retrieve_all():
            if _is_frozen(skill):
                sg = skill.get("status", {}).get("source_grid")
                if sg and len(sg) == 2:
                    return tuple(sg)
        return tuple(self.config.STRUCTURE_SHAPE)

    def _source_exp_name(self) -> str | None:
        for skill in self.skill_library.retrieve_all():
            if _is_frozen(skill):
                return skill.get("status", {}).get("source_exp")
        return None

    def _rebuild_parent_history_index(self):
        self._parent_history_index = {}
        self._body_by_hash = {}

        max_logged_label = -1
        generation_dirs = sorted(
            [p for p in self.logs_dir.glob("generation_*") if p.is_dir()],
            key=_generation_dir_key,
        )

        for gen_dir in generation_dirs:
            generation = _generation_dir_key(gen_dir)
            designs_path = gen_dir / "designs.json"
            if not designs_path.exists():
                continue

            try:
                with open(designs_path) as f:
                    designs = json.load(f)
            except Exception:
                continue

            for design in designs:
                body = design.get("body")
                fitness = design.get("fitness")
                if body is None or design.get("eval_status") == "suspicious":
                    continue

                body_hash = hashable(np.array(body, dtype=int))
                self._body_by_hash.setdefault(body_hash, body)
                if fitness is not None:
                    self._seen_designs.setdefault(body_hash, fitness)

                label = design.get("label")
                if isinstance(label, int):
                    max_logged_label = max(max_logged_label, label)

                parent_body_hash = design.get("parent_body_hash")
                if not parent_body_hash:
                    continue

                parent_body = self._body_by_hash.get(parent_body_hash)
                parent_fitness = self._seen_designs.get(parent_body_hash)
                if parent_body is None or parent_fitness is None or fitness is None:
                    continue

                entry = self._build_history_entry(
                    parent_body_hash=parent_body_hash,
                    parent_body=parent_body,
                    parent_fitness=parent_fitness,
                    child_body=body,
                    child_fitness=fitness,
                    proposal_path=design.get("proposal_path"),
                    based_on_skill=design.get("based_on_skill"),
                    generation=design.get("generation", generation),
                )
                self._parent_history_index.setdefault(parent_body_hash, []).append(entry)

        for entries in self._parent_history_index.values():
            entries.sort(key=lambda entry: entry["generation"])

        if max_logged_label >= self._label_counter:
            self._label_counter = max_logged_label + 1

    def _update_parent_history_index(self, results: list[dict]):
        new_entries_by_parent: dict[str, list[dict]] = {}

        for r in results:
            body = _body_to_list(r["body"])
            child_hash = hashable(np.array(body, dtype=int))
            self._body_by_hash.setdefault(child_hash, body)
            if r.get("fitness") is not None:
                self._seen_designs.setdefault(child_hash, r["fitness"])

            parent_body_hash = r.get("_parent_body_hash")
            parent_body = r.get("_parent_body")
            parent_fitness = r.get("_parent_fitness")
            if (
                not parent_body_hash
                or parent_body is None
                or parent_fitness is None
                or r.get("fitness") is None
                or r.get("eval_status") == "suspicious"
            ):
                continue

            entry = self._build_history_entry(
                parent_body_hash=parent_body_hash,
                parent_body=parent_body,
                parent_fitness=parent_fitness,
                child_body=body,
                child_fitness=r["fitness"],
                proposal_path=r.get("_proposal_path"),
                based_on_skill=r.get("based_on_skill"),
                generation=r.get("generation"),
            )
            new_entries_by_parent.setdefault(parent_body_hash, []).append(entry)

        for parent_body_hash, entries in new_entries_by_parent.items():
            history = self._parent_history_index.setdefault(parent_body_hash, [])
            history.extend(entries)
            history.sort(key=lambda entry: entry["generation"])

    def _build_history_entry(
        self,
        parent_body_hash: str,
        parent_body,
        parent_fitness: float,
        child_body,
        child_fitness: float,
        proposal_path: str | None,
        based_on_skill: str | None,
        generation: int | None,
    ) -> dict:
        voxel_diff = compute_voxel_diff(parent_body, child_body)
        return {
            "parent_body_hash": parent_body_hash,
            "parent_body": _body_to_list(parent_body),
            "parent_fitness": parent_fitness,
            "child_body": _body_to_list(child_body),
            "child_body_hash": hashable(np.array(child_body, dtype=int)),
            "child_fitness": child_fitness,
            "voxel_diff": voxel_diff,
            "proposal_path": proposal_path,
            "based_on_skill": based_on_skill,
            "generation": generation if generation is not None else -1,
        }

    def _format_parent_history_block(self, parent: dict) -> str:
        parent_body_hash = hashable(np.array(parent["body"], dtype=int))
        history = self._parent_history_index.get(parent_body_hash, [])
        if not history:
            return "No prior attempts on this exact parent."

        lines = []
        for entry in history:
            path = entry.get("proposal_path") or "?"
            if path == "A" and entry.get("based_on_skill"):
                path_label = f"[A skill={entry['based_on_skill']}]"
            else:
                path_label = f"[{path}]"
            diff_text = _format_voxel_diff(entry.get("voxel_diff", []))
            lines.append(
                f"Gen {entry.get('generation')} {path_label}\n"
                f"  child_fitness={entry.get('child_fitness', 0.0):.2f}\n"
                f"  voxel_diff={diff_text}\n"
                f"  child_body={_format_body(entry.get('child_body'))}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _body_to_list(body):
    return body.tolist() if hasattr(body, "tolist") else body


def _generation_dir_key(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return -1


def _format_skill_assignments_block(slot_assignments: list[dict], l1_only: bool = False) -> str:
    blocks = []
    for assignment in slot_assignments:
        slot_index = assignment["slot_index"]
        skill = assignment.get("skill")
        if not skill:
            blocks.append(
                f"[slot_index={slot_index}] based_on_skill=null\n"
                "  No active task-relevant skill is assigned for this slot.\n"
                "  Make a diverse but still parent-aware mutation."
            )
            continue
        if l1_only:
            blocks.append(
                f"[slot_index={slot_index}] based_on_skill={skill.get('skill_id', 'unknown')}\n"
                f"  L1 condition: {_skill_condition(skill) or 'N/A'}"
            )
        else:
            blocks.append(
                f"[slot_index={slot_index}] based_on_skill={skill.get('skill_id', 'unknown')}\n"
                f"  L1 condition: {_skill_condition(skill) or 'N/A'}\n"
                f"  L2 positive rules:\n{_format_l2_rules_for_llm(_skill_positive_rules(skill), 'positive')}\n"
                f"  L2 avoid rules:\n{_format_l2_rules_for_llm(_skill_negative_rules(skill), 'negative')}"
            )
    return "\n\n".join(blocks)


def _format_voxel_diff(voxel_diff: list[tuple]) -> str:
    if not voxel_diff:
        return "[]"
    return "[" + ", ".join(
        f"({row},{col}):{parent_val}->{child_val}"
        for row, col, parent_val, child_val in voxel_diff
    ) + "]"

def _format_body(body) -> str:
    """Format a body grid as a compact single-line string."""
    if hasattr(body, "tolist"):
        body = body.tolist()
    return "[" + ",".join(
        "[" + ",".join(str(cell) for cell in row) + "]"
        for row in body
    ) + "]"


def _format_generation_results(results: list[dict]) -> str:
    lines = []
    for r in sorted(results, key=lambda x: x["fitness"], reverse=True):
        body = r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"]
        skill_tag = f" [based_on_skill={r['based_on_skill']}]" if r.get("based_on_skill") else ""
        lines.append(f"Label={r['label']} Fitness={r['fitness']:.3f}{skill_tag}: body={body}")
    return "\n".join(lines)


def _format_generation_results_no_skill(results: list[dict]) -> str:
    """Format generation results without based_on_skill tags.

    Used by diagnosis prompts to let the LLM judge structural patterns
    purely from body grids and fitness, without attribution bias.
    """
    lines = []
    for r in sorted(results, key=lambda x: x["fitness"], reverse=True):
        body = r["body"].tolist() if hasattr(r["body"], "tolist") else r["body"]
        lines.append(f"Label={r['label']} Fitness={r['fitness']:.3f}: body={body}")
    return "\n".join(lines)


def _format_transfer_context_block(
    static_active: bool,
    same_grid: bool,
    elite_inject: bool,
    source_grid: tuple[int, int],
    current_grid: tuple[int, int],
    source_exp: str,
    current_env: str,
) -> str:
    """Render the top-level Transfer Context block.

    See spec §3.3.1-3.3.3. Returns empty string when static_active is False
    (i.e., normal phase, or pure cold-start where no transfer happened).
    """
    if not static_active:
        return ""

    src_g = f"{source_grid[0]}x{source_grid[1]}"
    cur_g = f"{current_grid[0]}x{current_grid[1]}"

    if same_grid:
        block = _TC_SAME_GRID_BASE.format(
            current_env=current_env,
            source_grid=src_g,
            source_exp=source_exp,
        )
        if elite_inject:
            block += _TC_SAME_GRID_ELITE_ADDENDUM
    else:
        block = _TC_CROSS_GRID_BASE.format(
            current_env=current_env,
            current_grid=cur_g,
            source_grid=src_g,
            source_exp=source_exp,
        )
        if elite_inject:
            block += _TC_CROSS_GRID_ELITE_ADDENDUM.format(source_grid=src_g)
    return block


def _format_static_reference_block(
    static_active: bool,
    elite_inject: bool,
    elites: list[dict],
) -> str:
    """Render the elite design reference block for PROPOSE_MUTATE.

    Returns empty string when:
    - not in warm phase (static_active=False), or
    - elite injection disabled (elite_inject=False), or
    - no elites available.
    """
    if not (static_active and elite_inject) or not elites:
        return ""
    lines = ["Static reference designs (top-fitness exemplars from source experiment):"]
    for e in elites:
        body = e.get("body")
        fitness = e.get("fitness", 0.0)
        label = e.get("label", "?")
        lines.append(f"  [{label} fitness={float(fitness):.3f}] body={body}")
    return "\n".join(lines) + "\n"


