import json
import re
from pathlib import Path

from solo_leveling.json_io import dump_json
from solo_leveling.llm_client import call_llm
from solo_leveling.prompts import (
    DIAGNOSE_POSITIVE_PROMPT,
    DIAGNOSE_NEGATIVE_PROMPT,
    CONSOLIDATE_CLUSTER_PROMPT,
)
from solo_leveling.config import LLM_MODEL_STRONG

# Skill field constraints — single source of truth
MAX_CONDITION_WORDS = 200

# Skill origin tags — only "imported" skills are frozen against mutation.
ORIGIN_NATIVE = "native"
ORIGIN_IMPORTED = "imported"


def _is_frozen(skill: dict) -> bool:
    """Return True iff the skill is an imported (frozen, read-only) skill."""
    status = skill.get("status")
    if not isinstance(status, dict):
        return False
    return status.get("origin") == ORIGIN_IMPORTED


def _validate_skill(skill: dict) -> dict:
    """Enforce field constraints on a skill dict before saving.

    Called by _save() so every write path (add, merge, diagnose) is covered.
    Truncates silently with a warning print.
    """
    skill_id = skill.get("skill_id", "unknown")

    # condition: max words. v1.5 stores the canonical condition in l1, but
    # legacy flat skills may still carry top-level condition.
    condition = _skill_condition(skill)
    if condition:
        words = condition.split()
        if len(words) > MAX_CONDITION_WORDS:
            print(f"[Validate] Truncating condition for '{skill_id}': {len(words)} → {MAX_CONDITION_WORDS} words")
            truncated = " ".join(words[:MAX_CONDITION_WORDS])
            skill.setdefault("l1", {})["condition"] = truncated
            if "condition" in skill:
                skill["condition"] = truncated

    return skill


def _hashable_list(body) -> str:
    """Hash a body grid (list[list[int]] or np.ndarray) without requiring numpy."""
    if hasattr(body, "tolist"):
        body = body.tolist()
    return "".join(str(int(v)) for row in body for v in row)


def _ensure_status(skill: dict) -> dict:
    """Normalize lifecycle/runtime metadata into the status field."""
    status = skill.get("status")
    if not isinstance(status, dict):
        status = {}

    skill.pop("retired_from", None)

    legacy_defaults = {
        "generation_born": None,
        "generation_updated": None,
        "total_samples": 0,
        "avg_gain": 0.0,
        "last_l2_diagnosed_generation": -1,
    }
    for key, default in legacy_defaults.items():
        if key in skill:
            status.setdefault(key, skill.pop(key))
        else:
            status.setdefault(key, default)

    legacy_absorbed = skill.pop("absorbed_skill_ids", None)
    merged_absorbed = []
    if isinstance(status.get("absorbed_skill_ids"), list):
        merged_absorbed.extend(status["absorbed_skill_ids"])
    if isinstance(legacy_absorbed, list):
        merged_absorbed.extend(legacy_absorbed)
    status["absorbed_skill_ids"] = [
        sid for sid in _unique_in_order(merged_absorbed)
        if sid and sid != skill.get("skill_id")
    ]

    # Version is removed from the new schema; silently discard if present.
    skill.pop("version", None)
    skill["status"] = status
    return status


def _infer_structure(skill: dict, condition: str) -> str:
    """Best-effort structure label for v1.5 L1 when loading legacy skills."""
    l1 = skill.get("l1")
    if isinstance(l1, dict) and l1.get("structure"):
        return str(l1["structure"])
    skill_id = skill.get("skill_id", "structure")
    base = re.sub(r"_v\d+(?:\.\d+)?$", "", str(skill_id))
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    if base:
        return base
    words = [w.strip(".,;:()[]{}").lower() for w in condition.split()]
    return next((w for w in words if w), "structure")


def _skill_condition(skill: dict) -> str:
    """Return canonical L1 condition with legacy flat-skill fallback."""
    l1 = skill.get("l1")
    if isinstance(l1, dict) and l1.get("condition"):
        return l1["condition"]
    return skill.get("condition", "")


def _rule_text(rule: dict, fallback_key: str) -> str:
    claim = rule.get("claim")
    description = rule.get("description") or rule.get(fallback_key) or ""
    if claim and description:
        return f"{claim}: {description}"
    return description or claim or ""


def _skill_positive_rules(skill: dict) -> list[dict]:
    """Return compact positive L2 rules, deriving them from legacy observations if needed."""
    l2 = skill.get("l2")
    if isinstance(l2, dict) and isinstance(l2.get("positive"), list):
        if l2["positive"] or not _legacy_positive_observations(skill):
            return l2["positive"]
    rules = []
    for obs in _legacy_positive_observations(skill)[:5]:
        claim = obs.get("what_worked")
        if claim:
            rules.append({
                "claim": _short_claim(claim),
                "description": claim,
            })
    return rules


def _skill_negative_rules(skill: dict) -> list[dict]:
    """Return compact negative L2 rules, deriving them from legacy observations if needed."""
    l2 = skill.get("l2")
    if isinstance(l2, dict) and isinstance(l2.get("negative"), list):
        if l2["negative"] or not _legacy_negative_observations(skill):
            return l2["negative"]
    rules = []
    for obs in _legacy_negative_observations(skill)[:5]:
        claim = obs.get("what_failed")
        if claim:
            rules.append({
                "claim": _short_claim(claim),
                "description": claim,
            })
    return rules


def _short_claim(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:4]
    return "_".join(words) if words else "observed_pattern"


def _normalize_rule(rule: dict, fallback_key: str) -> dict | None:
    text = _rule_text(rule, fallback_key)
    if not text:
        return None
    return {
        "leaf_id": rule.get("leaf_id"),
        "claim": rule.get("claim") or _short_claim(text),
        "description": rule.get("description") or rule.get(fallback_key) or text,
        "obs_ids": list(rule.get("obs_ids", [])),
        "birth_obs_count": int(rule.get("birth_obs_count", len(rule.get("obs_ids", [])))),
        "avg_gain": float(rule.get("avg_gain", 0.0)),
    }


def _normalize_l2_rules(rules: list[dict], polarity: str, start_index: int = 0) -> tuple[list[dict], int]:
    normalized_rules = []
    next_index = start_index
    prefix = "pos" if polarity == "positive" else "neg"
    for rule in rules:
        normalized = _normalize_rule(rule, "description")
        if not normalized:
            continue
        if not normalized.get("leaf_id"):
            normalized["leaf_id"] = f"{prefix}_{next_index}"
            next_index += 1
        else:
            match = re.search(r"_(\d+)$", str(normalized["leaf_id"]))
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)
        normalized.setdefault("obs_ids", [])
        normalized.setdefault("birth_obs_count", len(normalized["obs_ids"]))
        normalized.setdefault("avg_gain", 0.0)
        normalized_rules.append(normalized)
    return normalized_rules, next_index


def _normalize_skill(skill: dict) -> dict:
    """Normalize v1 legacy and v1.5 skills into the v1.5 shape."""
    legacy_positive_observations = _legacy_positive_observations(skill)
    legacy_negative_observations = _legacy_negative_observations(skill)
    condition = _skill_condition(skill)
    l1 = skill.get("l1")
    if not isinstance(l1, dict):
        l1 = {}
    l1.setdefault("structure", _infer_structure(skill, condition))
    l1.setdefault("condition", condition)
    l1.pop("specifier", None)
    skill["l1"] = l1

    l2 = skill.get("l2")
    if not isinstance(l2, dict):
        l2 = {}

    positive_rules = []
    raw_positive_rules = l2.get("positive", []) if isinstance(l2.get("positive"), list) else []
    for rule in raw_positive_rules:
        normalized = _normalize_rule(rule, "what_worked")
        if normalized:
            positive_rules.append(normalized)
    if not positive_rules:
        positive_rules = _skill_positive_rules(skill)

    negative_rules = []
    raw_negative_rules = l2.get("negative", []) if isinstance(l2.get("negative"), list) else []
    for rule in raw_negative_rules:
        normalized = _normalize_rule(rule, "what_failed")
        if normalized:
            negative_rules.append(normalized)
    if not negative_rules:
        negative_rules = _skill_negative_rules(skill)

    positive_rules, next_leaf_id = _normalize_l2_rules(positive_rules, "positive", 0)
    negative_rules, next_leaf_id = _normalize_l2_rules(negative_rules, "negative", next_leaf_id)

    skill["l2"] = {
        "positive": positive_rules,
        "negative": negative_rules,
        "next_leaf_id_counter": max(
            int(l2.get("next_leaf_id_counter", 0) or 0),
            next_leaf_id,
        ),
    }

    l3 = skill.get("l3")
    if not isinstance(l3, dict):
        l3 = {}
    first_positive_leaf_id = positive_rules[0].get("leaf_id") if positive_rules else None
    first_negative_leaf_id = negative_rules[0].get("leaf_id") if negative_rules else None

    l3_observations = [_clean_l3_observation(obs) for obs in l3.get("observations", [])]
    if not l3_observations:
        legacy_observations = skill.get("observations")
        if isinstance(legacy_observations, dict):
            l3_observations.extend(
                _clean_l3_observation(obs, default_leaf_id=first_positive_leaf_id)
                for obs in legacy_observations.get("positive", [])
            )
            l3_observations.extend(
                _clean_l3_observation(obs, default_leaf_id=first_negative_leaf_id)
                for obs in legacy_observations.get("negative", [])
            )
            if "observations" in legacy_observations and isinstance(legacy_observations["observations"], list):
                l3_observations.extend(_clean_l3_observation(obs) for obs in legacy_observations["observations"])
        l3_observations.extend(
            _clean_l3_observation(obs, default_leaf_id=first_positive_leaf_id)
            for obs in legacy_positive_observations
        )
        l3_observations.extend(
            _clean_l3_observation(obs, default_leaf_id=first_negative_leaf_id)
            for obs in legacy_negative_observations
        )

    next_obs_id = int(l3.get("next_obs_id_counter", 0) or 0)
    for obs in l3_observations:
        if "obs_id" not in obs:
            obs["obs_id"] = next_obs_id
            next_obs_id += 1
        else:
            next_obs_id = max(next_obs_id, int(obs["obs_id"]) + 1)
        obs.setdefault("parent_fitness", None)
        obs.setdefault("parent_body_hash", None)
        obs.setdefault("gain", None)
        obs.setdefault("used_leaf_id", None)
        obs.setdefault("source", "path_a")
        obs.setdefault("diagnosis_attempts", 0)
    l3["observations"] = l3_observations
    l3["next_obs_id_counter"] = max(
        int(l3.get("next_obs_id_counter", 0) or 0),
        next_obs_id,
    )
    skill["l3"] = l3

    # Keep the legacy mirror for older saved libraries and callers that still
    # inspect top-level `condition`; all prompt rendering uses l1/l2 helpers.
    if l1.get("condition"):
        skill["condition"] = l1["condition"]
    skill.pop("observations", None)
    skill.pop("positive_observations", None)
    skill.pop("negative_observations", None)
    _ensure_status(skill)
    _refresh_skill_stats(skill)
    return skill


def _legacy_positive_observations(skill: dict) -> list[dict]:
    observations = skill.get("positive_observations")
    return observations if isinstance(observations, list) else []


def _legacy_negative_observations(skill: dict) -> list[dict]:
    observations = skill.get("negative_observations")
    return observations if isinstance(observations, list) else []


def _clean_l3_observation(obs: dict, default_leaf_id: str | None = None) -> dict:
    cleaned = dict(obs)
    cleaned.pop("what_worked", None)
    cleaned.pop("what_failed", None)
    if default_leaf_id and not cleaned.get("used_leaf_id"):
        cleaned["used_leaf_id"] = default_leaf_id
    # v2 fields — backfill defaults for legacy data
    cleaned.setdefault("source", "path_a")        # legacy v1.5 was all LLM-proposed
    cleaned.setdefault("diagnosis_attempts", 0)
    return cleaned


def _refresh_skill_stats(skill: dict):
    all_observations = skill.get("l3", {}).get("observations", [])
    gains = [obs.get("gain") for obs in all_observations if obs.get("gain") is not None]
    _set_status(skill, "total_samples", len(all_observations))
    _set_status(skill, "avg_gain", sum(gains) / len(gains) if gains else 0.0)
    # v2: positive_rate — fraction of obs with gain > 0 (informational; complements avg_gain)
    n_positive = sum(1 for g in gains if g > 0)
    _set_status(skill, "positive_rate", n_positive / len(gains) if gains else 0.0)


def _observations_for_polarity(skill: dict, polarity: str) -> list[dict]:
    bucket = "positive" if polarity == "positive" else "negative"
    leaf_ids = {
        leaf.get("leaf_id")
        for leaf in skill.get("l2", {}).get(bucket, [])
        if leaf.get("leaf_id")
    }
    observations = []
    for obs in skill.get("l3", {}).get("observations", []):
        if obs.get("used_leaf_id") in leaf_ids:
            observations.append(obs)
    return observations


def _append_l2_rule(
    skill: dict,
    polarity: str,
    rule: dict | None,
    fallback_text: str | None = None,
    obs_id: int | None = None,
    gain: float | None = None,
) -> str | None:
    """Append or update an L2 leaf and link the supporting observation."""
    if not rule and fallback_text:
        rule = {
            "claim": _short_claim(fallback_text),
            "description": fallback_text,
        }
    normalized = _normalize_rule(rule or {}, "description")
    if not normalized:
        return None

    l2 = skill.setdefault("l2", {})
    l2.setdefault("positive", [])
    l2.setdefault("negative", [])
    next_leaf_id = int(l2.get("next_leaf_id_counter", 0) or 0)
    bucket = l2["positive" if polarity == "positive" else "negative"]
    normalized_claim = normalized["claim"].strip().lower()
    for existing in bucket:
        existing_claim = str(existing.get("claim", "")).strip().lower()
        if existing_claim == normalized_claim:
            _link_observation_to_rule(existing, obs_id, gain)
            return existing.get("leaf_id")
    prefix = "pos" if polarity == "positive" else "neg"
    normalized["leaf_id"] = normalized.get("leaf_id") or f"{prefix}_{next_leaf_id}"
    normalized["obs_ids"] = list(normalized.get("obs_ids", []))
    normalized["birth_obs_count"] = int(normalized.get("birth_obs_count", 0) or 0)
    normalized["avg_gain"] = float(normalized.get("avg_gain", 0.0) or 0.0)
    _link_observation_to_rule(normalized, obs_id, gain)
    bucket.append(normalized)
    l2["next_leaf_id_counter"] = next_leaf_id + 1
    return normalized["leaf_id"]


def _link_observation_to_rule(rule: dict, obs_id: int | None, gain: float | None):
    if obs_id is None:
        return
    obs_ids = rule.setdefault("obs_ids", [])
    if obs_id in obs_ids:
        return
    old_count = len(obs_ids)
    old_avg = float(rule.get("avg_gain", 0.0) or 0.0)
    obs_ids.append(obs_id)
    rule.setdefault("birth_obs_count", min(old_count, int(rule.get("birth_obs_count", old_count) or old_count)))
    if gain is not None:
        new_count = old_count + 1
        rule["avg_gain"] = old_avg + (float(gain) - old_avg) / new_count


def _format_l2_rules_for_llm(rules: list[dict], polarity: str, max_items: int = 5) -> str:
    """Render compact L2 rules for prompts."""
    if not rules:
        return f"  (no {polarity} L2 rules yet)"
    lines = []
    for rule in rules[:max_items]:
        claim = rule.get("claim", "unnamed_rule")
        description = rule.get("description", "")
        # Prefer real accumulated obs_ids length; fall back to frozen birth_obs_count
        # when obs_ids is empty (imported skill case where obs_ids was stripped).
        obs_ids = rule.get("obs_ids") or []
        if obs_ids:
            obs_count = len(obs_ids)
        else:
            obs_count = int(rule.get("birth_obs_count", 0) or 0)
        avg_gain = rule.get("avg_gain", 0.0)
        leaf_id = rule.get("leaf_id", "?")
        lines.append(
            f"  - leaf_id={leaf_id}, claim={claim} (n={obs_count}, avg_gain={float(avg_gain):.3f}): {description}"
        )
    return "\n".join(lines)


def _get_status(skill: dict, key: str, default=None):
    status = skill.get("status")
    if isinstance(status, dict) and key in status:
        return status.get(key, default)
    if key in skill:
        return skill.get(key, default)
    return default


def _set_status(skill: dict, key: str, value):
    status = _ensure_status(skill)
    status[key] = value


def _unique_in_order(items) -> list:
    """Deduplicate a flat iterable while preserving first-seen order."""
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _get_absorbed_skill_ids(skill: dict) -> list[str]:
    absorbed_ids = _get_status(skill, "absorbed_skill_ids", [])
    if not isinstance(absorbed_ids, list):
        return []
    skill_id = skill.get("skill_id")
    return [sid for sid in _unique_in_order(absorbed_ids) if sid and sid != skill_id]


def _observation_count(skill: dict) -> int:
    return len(skill.get("l3", {}).get("observations", []))


def _choose_canonical_skill(skills: list[dict]) -> dict:
    """Pick a stable canonical representative for a duplicate cluster."""
    if not skills:
        raise ValueError("Cannot choose a canonical skill from an empty list")

    def sort_key(skill: dict):
        generation_born = _get_status(skill, "generation_born")
        if generation_born is None:
            generation_born = float("inf")
        return (
            -_observation_count(skill),
            -len(skill.get("task_family", [])),
            generation_born,
            skill.get("skill_id", ""),
        )

    return min(skills, key=sort_key)


def _drop_legacy_fields(skill: dict):
    """Remove fields that no longer belong to the active skill schema."""
    for key in (
        "explanation",
        "what_works",
        "what_fails",
        "negative_streak",
        "retired_from",
        "absorbed_skill_ids",
        "design_templates",
        "design_template",
    ):
        skill.pop(key, None)


def _build_merge_reasoning(
    cluster: dict,
    canonical: dict,
    absorbed_ids: list[str],
    cluster_skills: list[dict],
) -> dict:
    """Create audit-only reasoning for a weak merge event."""
    canonical_id = canonical.get("skill_id")
    canonical_obs = _observation_count(canonical)
    canonical_tasks = len(canonical.get("task_family", []))
    canonical_born = _get_status(canonical, "generation_born")
    return {
        "merge_type": "weak_merge",
        "why_merge": cluster.get("reason", ""),
        "canonical_skill_id": canonical_id,
        "absorbed_skill_ids": absorbed_ids,
        "cluster_skill_ids": [s.get("skill_id") for s in cluster_skills],
        "canonical_selection_basis": (
            "Deterministic canonical selection: prefer more total observations, "
            "then broader task_family, then earlier generation_born, then skill_id."
        ),
        "canonical_snapshot": {
            "observation_count": canonical_obs,
            "task_family_size": canonical_tasks,
            "generation_born": canonical_born,
        },
    }


class SkillLibrary:
    def __init__(self, skills_dir: str, static_active_gens: int | None = None):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._changed_since_last_consolidation = False
        self._static_active_gens = static_active_gens

    def mark_changed(self):
        self._changed_since_last_consolidation = True

    def has_pending_changes(self) -> bool:
        return self._changed_since_last_consolidation

    def clear_change_flag(self):
        self._changed_since_last_consolidation = False

    def _load_all(self) -> list[dict]:
        skills = []
        for path in self.skills_dir.glob("*.json"):
            if path.name.startswith("_"):
                continue   # underscore-prefix files are library metadata, not skills
            try:
                with open(path) as f:
                    data = json.load(f)
                if "skill_id" not in data:
                    print(f"[SkillLibrary] Skipping {path}: missing 'skill_id'")
                    continue
                skills.append(_normalize_skill(data))
            except Exception as e:
                print(f"[SkillLibrary] Failed to load {path}: {e}")
        return skills

    def _save(self, skill: dict):
        skill = _normalize_skill(skill)
        skill = _validate_skill(skill)
        _ensure_status(skill)
        _drop_legacy_fields(skill)
        l3 = skill.setdefault("l3", {})
        l3.setdefault("observations", [])
        l3.setdefault("next_obs_id_counter", 0)
        skill.pop("observations", None)
        skill.pop("positive_observations", None)
        skill.pop("negative_observations", None)
        skill_id = skill["skill_id"]
        # Sanitize filename — use hash suffix to avoid collisions between
        # skill_ids that differ only in characters removed by sanitization.
        safe_id = re.sub(r"[^\w\-.]", "_", skill_id)
        path = self.skills_dir / f"{safe_id}.json"
        # Guard: if file exists but belongs to a different skill_id, append hash
        if path.exists():
            try:
                with open(path) as f:
                    existing = json.load(f)
                if existing.get("skill_id") != skill_id:
                    import hashlib
                    h = hashlib.md5(skill_id.encode()).hexdigest()[:6]
                    path = self.skills_dir / f"{safe_id}_{h}.json"
            except Exception:
                pass
        dump_json(skill, path)

    # ──────────────────────────────────────────────────────────
    # v2: library-level metadata (global obs counter + unassigned pool)
    # ──────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self.skills_dir / "_library_meta.json"

    def _load_meta(self) -> dict:
        path = self._meta_path()
        if not path.exists():
            return {
                "schema_version": "v2",
                "next_obs_id_counter": self._estimate_initial_counter(),
                "unassigned_pool": [],
            }
        try:
            with open(path) as f:
                meta = json.load(f)
            meta.setdefault("schema_version", "v2")
            meta.setdefault("next_obs_id_counter", self._estimate_initial_counter())
            meta.setdefault("unassigned_pool", [])
            return meta
        except Exception as e:
            print(f"[SkillLibrary] Failed to load meta {path}: {e}; using defaults")
            return {
                "schema_version": "v2",
                "next_obs_id_counter": self._estimate_initial_counter(),
                "unassigned_pool": [],
            }

    def _save_meta(self, meta: dict) -> None:
        dump_json(meta, self._meta_path())

    def _estimate_initial_counter(self) -> int:
        """Walk all loaded skills' L3 observations to find max obs_id; return max+1.

        Used on first v2 load when meta file doesn't exist yet. Note: this still
        leaves cross-skill ID collisions in legacy data; new obs from this point on
        are globally unique.
        """
        max_id = -1
        for skill in self._load_all():
            for obs in skill.get("l3", {}).get("observations", []):
                oid = obs.get("obs_id")
                if isinstance(oid, int) and oid > max_id:
                    max_id = oid
        return max_id + 1

    def get_next_obs_id(self) -> int:
        """Allocate one globally-unique obs_id and persist counter."""
        meta = self._load_meta()
        oid = meta["next_obs_id_counter"]
        meta["next_obs_id_counter"] = oid + 1
        self._save_meta(meta)
        return oid

    def append_to_pool(self, pool_obs: dict) -> None:
        """Add an unassigned obs to the library-level pool."""
        meta = self._load_meta()
        meta["unassigned_pool"].append(pool_obs)
        self._save_meta(meta)

    def get_pool(self) -> list[dict]:
        return self._load_meta()["unassigned_pool"]

    def remove_from_pool(self, obs_ids: list[int]) -> list[dict]:
        """Remove obs from pool by obs_id; return the removed ones."""
        meta = self._load_meta()
        keep, removed = [], []
        for o in meta["unassigned_pool"]:
            if o.get("obs_id") in set(obs_ids):
                removed.append(o)
            else:
                keep.append(o)
        meta["unassigned_pool"] = keep
        self._save_meta(meta)
        return removed

    def _delete_skill(self, skill_id: str):
        """Remove a skill's JSON file from disk (used after consolidation merge)."""
        safe_id = re.sub(r"[^\w\-.]", "_", skill_id)
        path = self.skills_dir / f"{safe_id}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("skill_id") == skill_id:
                    path.unlink()
                    return
            except Exception:
                pass
        # Also try hash-suffixed variants
        for p in self.skills_dir.glob(f"{safe_id}_*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                if data.get("skill_id") == skill_id:
                    p.unlink()
                    return
            except Exception:
                pass

    def retrieve(
        self,
        task_name: str,
        current_gen: int | None = None,
        for_writing: bool = False,
    ) -> list[dict]:
        """Return skills matching task_name (no sorting, no truncation).

        Filtering rules (in order of precedence):
          - for_writing=True: always exclude imported (writes never go to imported)
          - current_gen given AND self._static_active_gens not None: phase filter
            - gen < K  -> imported only (warm phase)
            - gen >= K -> native only (normal phase)
          - otherwise: just task_family match (legacy compat / cold-start)
        """
        skills = self._load_all()
        matched = [s for s in skills if task_name in s.get("task_family", [])]

        if for_writing:
            matched = [s for s in matched if not _is_frozen(s)]
        elif current_gen is not None and self._static_active_gens is not None:
            if current_gen < self._static_active_gens:
                matched = [s for s in matched if _is_frozen(s)]
            else:
                matched = [s for s in matched if not _is_frozen(s)]
        return matched

    def retrieve_all(self, include_fully_retired: bool = False) -> list[dict]:
        """Return all skills. By default excludes skills with empty task_family."""
        all_skills = self._load_all()
        if not include_fully_retired:
            all_skills = [s for s in all_skills if s.get("task_family", [])]
        return all_skills

    def expand_task_family(self, skill_id: str, task_name: str, generation: int, ops_log_path: str = None):
        """Add task_name to an existing skill's task_family (unretrieved diagnosis).

        Imported (frozen) skills are skipped — their task_family is set at import time
        via --retrieve-as-task and never modified afterwards.
        """
        all_skills = {s["skill_id"]: s for s in self._load_all()}
        if skill_id not in all_skills:
            print(f"[SkillLibrary] expand_task_family: skill {skill_id} not found")
            return
        skill = all_skills[skill_id]
        if _is_frozen(skill):
            print(f"[SkillLibrary] expand_task_family: skipping frozen imported skill '{skill_id}'")
            return
        _ensure_status(skill)
        if task_name in skill.get("task_family", []):
            return  # already included
        skill.setdefault("task_family", []).append(task_name)
        _set_status(skill, "generation_updated", generation)
        self._save(skill)
        self.mark_changed()
        print(f"[SkillLibrary] Expanded '{skill_id}' task_family with '{task_name}'")
        if ops_log_path:
            self._append_ops_log(ops_log_path, [{
                "generation": generation,
                "action": "expand_task_family",
                "skill_id": skill_id,
                "trigger": "unretrieved_diagnose",
            }])

    def diagnose(
        self,
        skill_id: str,
        positives: list[dict],
        negatives: list[dict],
        generation: int,
    ) -> dict | None:
        """Append L3 evidence and update L2 leaves.

        Does NOT rewrite L1 condition or store explanation text in L3.

        positives / negatives: lists of design dicts, each with keys
            {body, fitness, label, ...}. Both lists can be empty.

        Frozen imported skills are skipped (defense-in-depth; v2 pipeline has its own
        guard but this protects any caller still using the v1.5 path).

        Returns the updated skill dict (so caller can save it) or None if the
        skill was not found or nothing was appended.
        """
        if not positives and not negatives:
            return None

        all_skills = {s["skill_id"]: s for s in self._load_all()}
        if skill_id not in all_skills:
            print(f"[SkillLibrary] Diagnose: skill {skill_id} not found")
            return None

        skill = all_skills[skill_id]
        if _is_frozen(skill):
            print(f"[SkillLibrary] Diagnose: skipping frozen imported skill '{skill_id}'")
            return None
        _normalize_skill(skill)
        _ensure_status(skill)
        l3 = skill.setdefault("l3", {})
        l3.setdefault("observations", [])
        l3.setdefault("next_obs_id_counter", 0)

        any_appended = False

        # --- Positive direction ---
        if positives:
            existing_text = _format_observations_for_llm(
                _observations_for_polarity(skill, "positive"),
                "what_worked",
            )
            new_text = _format_new_designs_for_llm(positives)
            prompt = DIAGNOSE_POSITIVE_PROMPT.format(
                l1_condition=_skill_condition(skill) or "N/A",
                l2_positive=_format_l2_rules_for_llm(_skill_positive_rules(skill), "positive"),
                existing_positives=existing_text,
                new_designs=new_text,
            )
            try:
                response = call_llm([{"role": "user", "content": prompt}])
                for obs_result in response.get("observations", []):
                    idx = obs_result.get("body_index")
                    rule = obs_result.get("rule")
                    if rule is None or idx is None:
                        continue
                    if not isinstance(idx, int) or idx < 0 or idx >= len(positives):
                        continue
                    design = positives[idx]
                    body = design["body"]
                    if hasattr(body, "tolist"):
                        body = body.tolist()
                    obs_id = int(l3.get("next_obs_id_counter", 0) or 0)
                    l3["next_obs_id_counter"] = obs_id + 1
                    parent_fitness = design.get("parent_fitness")
                    gain = design.get("gain")
                    if gain is None and parent_fitness is not None:
                        gain = float(design["fitness"] - parent_fitness)
                    leaf_id = _append_l2_rule(
                        skill,
                        "positive",
                        rule,
                        obs_id=obs_id,
                        gain=gain,
                    )
                    if leaf_id is None:
                        continue
                    obs = {
                        "obs_id": obs_id,
                        "body": body,
                        "fitness": float(design["fitness"]),
                        "parent_body_hash": design.get("parent_body_hash"),
                        "parent_fitness": float(parent_fitness) if parent_fitness is not None else None,
                        "gain": float(gain) if gain is not None else None,
                        "used_leaf_id": leaf_id,
                        "gen_observed": generation,
                    }
                    l3["observations"].append(obs)
                    any_appended = True
            except Exception as e:
                print(f"[SkillLibrary] Diagnose positive failed for {skill_id}: {e}")

        # --- Negative direction ---
        if negatives:
            existing_text = _format_observations_for_llm(
                _observations_for_polarity(skill, "negative"),
                "what_failed",
            )
            new_text = _format_new_designs_for_llm(negatives)
            prompt = DIAGNOSE_NEGATIVE_PROMPT.format(
                l1_condition=_skill_condition(skill) or "N/A",
                l2_negative=_format_l2_rules_for_llm(_skill_negative_rules(skill), "negative"),
                existing_negatives=existing_text,
                new_designs=new_text,
            )
            try:
                response = call_llm([{"role": "user", "content": prompt}])
                for obs_result in response.get("observations", []):
                    idx = obs_result.get("body_index")
                    rule = obs_result.get("rule")
                    if rule is None or idx is None:
                        continue
                    if not isinstance(idx, int) or idx < 0 or idx >= len(negatives):
                        continue
                    design = negatives[idx]
                    body = design["body"]
                    if hasattr(body, "tolist"):
                        body = body.tolist()
                    obs_id = int(l3.get("next_obs_id_counter", 0) or 0)
                    l3["next_obs_id_counter"] = obs_id + 1
                    parent_fitness = design.get("parent_fitness")
                    gain = design.get("gain")
                    if gain is None and parent_fitness is not None:
                        gain = float(design["fitness"] - parent_fitness)
                    leaf_id = _append_l2_rule(
                        skill,
                        "negative",
                        rule,
                        obs_id=obs_id,
                        gain=gain,
                    )
                    if leaf_id is None:
                        continue
                    obs = {
                        "obs_id": obs_id,
                        "body": body,
                        "fitness": float(design["fitness"]),
                        "parent_body_hash": design.get("parent_body_hash"),
                        "parent_fitness": float(parent_fitness) if parent_fitness is not None else None,
                        "gain": float(gain) if gain is not None else None,
                        "used_leaf_id": leaf_id,
                        "gen_observed": generation,
                    }
                    l3["observations"].append(obs)
                    any_appended = True
            except Exception as e:
                print(f"[SkillLibrary] Diagnose negative failed for {skill_id}: {e}")

        if not any_appended:
            return None

        _set_status(skill, "generation_updated", generation)
        _refresh_skill_stats(skill)

        return skill

    def consolidate(self, task_name: str, generation: int,
                    ops_log_path: str = None) -> bool:
        """
        Step 1a: Cluster semantically similar skills via LLM.
        Step 1b: Canonicalize each duplicate cluster into one skill.
        Imported (frozen) skills are excluded from cluster candidates — they are read-only.
        Returns True if any merges happened.
        """
        skills = self.retrieve(task_name, for_writing=True)
        if len(skills) < 2:
            return False

        print(f"  [Consolidation] Scanning {len(skills)} skills for redundancy...")

        # Step 1a: Cluster via LLM
        skills_full_content = _format_skills_l1(skills)
        messages = [
            {"role": "user", "content": CONSOLIDATE_CLUSTER_PROMPT.format(
                skills_full_content=skills_full_content,
            )}
        ]
        try:
            cluster_response = call_llm(messages, model=LLM_MODEL_STRONG)
            clusters = cluster_response.get("clusters", [])
        except Exception as e:
            print(f"  [Consolidation] Clustering LLM call failed: {e}")
            return False

        # Filter to multi-skill clusters only
        merge_clusters = [c for c in clusters if len(c.get("skill_ids", [])) >= 2]
        if not merge_clusters:
            print(f"  [Consolidation] No redundant clusters found")
            return False

        print(f"  [Consolidation] Found {len(merge_clusters)} clusters to merge")

        # Build skill lookup
        skill_map = {s["skill_id"]: s for s in skills}
        any_merged = False
        log_records = []

        for cluster in merge_clusters:
            cluster_skill_ids = _unique_in_order(cluster.get("skill_ids", []))

            # Validate all skill_ids exist
            cluster_skills = [skill_map[sid] for sid in cluster_skill_ids if sid in skill_map]
            if len(cluster_skills) < 2:
                continue

            canonical = json.loads(json.dumps(_choose_canonical_skill(cluster_skills)))
            canonical_id = canonical["skill_id"]
            absorbed_ids = [s["skill_id"] for s in cluster_skills if s["skill_id"] != canonical_id]
            if not absorbed_ids:
                continue

            generation_born_candidates = [
                _get_status(s, "generation_born") for s in cluster_skills
                if _get_status(s, "generation_born") is not None
            ]
            merged_task_family = _unique_in_order(
                task for s in cluster_skills for task in s.get("task_family", [])
            )
            canonical["task_family"] = merged_task_family
            merged_l3_observations = []
            seen_obs_ids = set()
            obs_id_map = {}
            next_obs_id = 0
            for s in cluster_skills:
                for obs in s.get("l3", {}).get("observations", []):
                    obs_copy = dict(obs)
                    original_id = obs_copy.get("obs_id")
                    obs_key = (s.get("skill_id"), original_id)
                    if obs_key in seen_obs_ids:
                        continue
                    seen_obs_ids.add(obs_key)
                    obs_copy["obs_id"] = next_obs_id
                    obs_id_map[obs_key] = next_obs_id
                    next_obs_id += 1
                    merged_l3_observations.append(obs_copy)
            canonical["l3"] = {
                "observations": merged_l3_observations,
                "next_obs_id_counter": next_obs_id,
            }
            merged_l2_positive = []
            merged_l2_negative = []
            for s in cluster_skills:
                skill_id_for_obs = s.get("skill_id")
                for rule in _skill_positive_rules(s):
                    rule_copy = dict(rule)
                    rule_copy["obs_ids"] = [
                        obs_id_map[(skill_id_for_obs, obs_id)]
                        for obs_id in rule.get("obs_ids", [])
                        if (skill_id_for_obs, obs_id) in obs_id_map
                    ]
                    merged_l2_positive.append(rule_copy)
                for rule in _skill_negative_rules(s):
                    rule_copy = dict(rule)
                    rule_copy["obs_ids"] = [
                        obs_id_map[(skill_id_for_obs, obs_id)]
                        for obs_id in rule.get("obs_ids", [])
                        if (skill_id_for_obs, obs_id) in obs_id_map
                    ]
                    merged_l2_negative.append(rule_copy)
            canonical["l2"] = {"positive": [], "negative": []}
            for rule in merged_l2_positive:
                _append_l2_rule(canonical, "positive", rule)
            for rule in merged_l2_negative:
                _append_l2_rule(canonical, "negative", rule)
            canonical["condition"] = _skill_condition(canonical)
            canonical["reasoning"] = _build_merge_reasoning(
                cluster,
                canonical,
                absorbed_ids,
                cluster_skills,
            )
            _ensure_status(canonical)
            _drop_legacy_fields(canonical)
            _set_status(
                canonical,
                "generation_born",
                min(generation_born_candidates) if generation_born_candidates else generation,
            )
            _set_status(canonical, "generation_updated", generation)
            merged_absorbed_ids = _unique_in_order(
                sid
                for skill in cluster_skills
                for sid in ([skill.get("skill_id")] + _get_absorbed_skill_ids(skill))
                if sid and sid != canonical_id
            )
            _set_status(canonical, "origin", "native")
            _set_status(canonical, "absorbed_skill_ids", merged_absorbed_ids)

            self._save(canonical)

            for old_id in absorbed_ids:
                self._delete_skill(old_id)

            # Remove consumed skills from map so later clusters can't reuse them
            old_ids = [s["skill_id"] for s in cluster_skills]
            for old_id in old_ids:
                skill_map.pop(old_id, None)

            any_merged = True
            log_records.append({
                "generation": generation,
                "action": "consolidate_merge",
                "skill_id": canonical_id,
                "merged_from": absorbed_ids,
                "trigger": "consolidation",
            })
            print(f"  [Consolidation] Absorbed {absorbed_ids} into '{canonical_id}'")

        if ops_log_path and log_records:
            self._append_ops_log(ops_log_path, log_records)

        return any_merged

    def _append_ops_log(self, path: str, records: list[dict]):
        existing = []
        try:
            with open(path) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        existing.extend(records)
        dump_json(existing, path)


def _format_observations_for_llm(observations: list[dict], claim_key: str) -> str:
    """Render factual observations for DIAGNOSE prompts."""
    if not observations:
        return "  (none yet)"
    lines = []
    for obs in observations:
        body = obs.get("body")
        if hasattr(body, "tolist"):
            body = body.tolist()
        fitness = obs.get("fitness", 0.0)
        lines.append(
            f"  - obs_id={obs.get('obs_id')} | fitness={fitness:.3f} | "
            f"parent_body_hash={obs.get('parent_body_hash')} | "
            f"parent_fitness={obs.get('parent_fitness')} | gain={obs.get('gain')} | "
            f"used_leaf_id={obs.get('used_leaf_id')}\n"
            f"    body: {body}"
        )
    return "\n".join(lines)


def _format_new_designs_for_llm(designs: list[dict]) -> str:
    """Render new designs as an indexed list for DIAGNOSE prompts.

    The index is what the LLM uses to reference designs in its JSON response
    ({"body_index": N, ...}), so the agent can look up body/fitness/label later.
    """
    lines = []
    for i, d in enumerate(designs):
        body = d.get("body")
        if hasattr(body, "tolist"):
            body = body.tolist()
        fitness = d.get("fitness", 0.0)
        lines.append(f"  [index {i}] fitness={fitness:.3f}, body: {body}")
    return "\n".join(lines)


def _format_skills_full(skills: list[dict]) -> str:
    """Full skill content for consolidation and new-insight prompts."""
    if not skills:
        return ""
    entries = []
    for s in skills:
        s = _normalize_skill(dict(s))
        entries.append(
            f"[{s.get('skill_id', 'unknown')}]\n"
            f"  task_family: {s.get('task_family', [])}\n"
            f"  L1 structure: {s.get('l1', {}).get('structure', 'N/A')}\n"
            f"  L1 condition: {_skill_condition(s) or 'N/A'}\n"
            f"  L2 positive rules:\n{_format_l2_rules_for_llm(_skill_positive_rules(s), 'positive')}\n"
            f"  L2 avoid rules:\n{_format_l2_rules_for_llm(_skill_negative_rules(s), 'negative')}"
        )
    return "\n\n".join(entries)


def _format_skills_l1(skills: list[dict]) -> str:
    """Render only L1 identities for consolidation prompts."""
    if not skills:
        return ""
    entries = []
    for s in skills:
        s = _normalize_skill(dict(s))
        entries.append(
            f"[{s.get('skill_id', 'unknown')}]\n"
            f"  task_family: {s.get('task_family', [])}\n"
            f"  L1 structure: {s.get('l1', {}).get('structure', 'N/A')}\n"
            f"  L1 condition: {_skill_condition(s) or 'N/A'}"
        )
    return "\n\n".join(entries)


