"""L1 New Insight stage (§5): create new skills from unassigned pool / fallback."""
import json
from solo_leveling.llm_client import call_llm
from solo_leveling.prompts import NEW_INSIGHT_PROMPT
from solo_leveling.skill_library import SkillLibrary, _ensure_status, _set_status
from solo_leveling.attribution import _build_skills_block

NEW_INSIGHT_FALLBACK_TOP_K = 6
NEW_INSIGHT_FALLBACK_BOTTOM_K = 6


def run_new_insight(
    library: SkillLibrary,
    current_gen: int,
    current_gen_obs_in_skills: list[dict],
    task_name: str,
) -> dict | None:
    """Stage 2 (§5): possibly create one new skill. Returns the new skill dict or None.

    `current_gen_obs_in_skills` is the list of L3 obs across all skills with gen_observed == current_gen
    — used as fallback candidate pool when unassigned_pool has no positive obs.
    """
    pool = library.get_pool()
    pool_pos = [o for o in pool if (o.get("gain") or 0.0) > 0]
    pool_neg = [o for o in pool if (o.get("gain") or 0.0) <= 0]

    if pool_pos:
        high = pool_pos
        low = pool_neg
        source = "pool"
    else:
        gen_pos = [o for o in current_gen_obs_in_skills if (o.get("gain") or 0.0) > 0]
        if not gen_pos:
            return None  # skip: no positive candidates anywhere
        high = sorted(gen_pos, key=lambda o: -(o.get("gain") or 0.0))[:NEW_INSIGHT_FALLBACK_TOP_K]
        low = sorted(current_gen_obs_in_skills, key=lambda o: (o.get("gain") or 0.0))[:NEW_INSIGHT_FALLBACK_BOTTOM_K]
        source = "fallback"

    skills_block = _build_skills_block(library.retrieve(task_name, for_writing=True))
    high_block = _format_obs_block(high)
    low_block = _format_obs_block(low)

    prompt = NEW_INSIGHT_PROMPT.format(
        task_name=task_name,
        high_designs=high_block,
        low_designs=low_block,
        existing_skills_block=skills_block,
        # legacy templates may use {low_skill_hint}; supply empty
        low_skill_hint="",
    )

    try:
        response = call_llm([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"[NewInsight] LLM call failed: {e}")
        return None

    decision = response.get("decision", {})
    if decision.get("action") != "add" or decision.get("skill") is None:
        return None

    skill_data = decision["skill"]
    inspired_ids = decision.get("inspired_obs_ids", []) or []
    new_skill = _build_new_skill(skill_data, current_gen, decision.get("reasoning"))

    # Migrate inspired pool obs only (Invariant 1)
    pool_index = {o.get("obs_id"): o for o in pool}
    inspired_in_pool = [oid for oid in inspired_ids if oid in pool_index]
    for oid in inspired_in_pool:
        pool_obs = pool_index[oid]
        new_skill.setdefault("l3", {}).setdefault("observations", []).append(_pool_to_l3(pool_obs))
    if inspired_in_pool:
        library.remove_from_pool(inspired_in_pool)

    library._save(new_skill)
    library.mark_changed()
    return new_skill


def _format_obs_block(obs_list: list[dict]) -> str:
    """Render obs list as JSON for prompt injection."""
    return json.dumps(
        [
            {"obs_id": o.get("obs_id"), "body": o.get("body"), "gain": o.get("gain")}
            for o in obs_list
        ],
        indent=2,
    )


def _pool_to_l3(pool_obs: dict) -> dict:
    return {
        "obs_id": pool_obs.get("obs_id"),
        "body": pool_obs.get("body"),
        "fitness": pool_obs.get("fitness"),
        "parent_body_hash": pool_obs.get("parent_body_hash"),
        "parent_fitness": pool_obs.get("parent_fitness"),
        "gain": pool_obs.get("gain"),
        "used_leaf_id": None,
        "gen_observed": pool_obs.get("gen_observed"),
        "source": "path_g",
        "diagnosis_attempts": 0,
        # Preserve frozen-path-a provenance for audit trail
        "source_origin": pool_obs.get("source_origin"),
        "source_imported_skill_id": pool_obs.get("source_imported_skill_id"),
        "source_leaf_id": pool_obs.get("source_leaf_id"),
    }


def _build_new_skill(skill_data: dict, current_gen: int, reasoning: dict | None) -> dict:
    skill = dict(skill_data)
    skill.setdefault("l2", {"positive": [], "negative": [], "next_leaf_id_counter": 0})
    skill.setdefault("l3", {"observations": [], "next_obs_id_counter": 0})
    if reasoning:
        skill["reasoning"] = dict(reasoning)
    _ensure_status(skill)
    _set_status(skill, "origin", "native")
    _set_status(skill, "born_via", "new_insight")
    _set_status(skill, "generation_born", current_gen)
    _set_status(skill, "generation_updated", current_gen)
    _set_status(skill, "last_l2_diagnosed_generation", -1)
    return skill
