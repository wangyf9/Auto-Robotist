"""Attribution stage (§4): route evaluated obs to skills or unassigned_pool.

Path A (with based_on_skill marker): rule-based, no LLM, trust-marker (§7.2).
Path G (no marker): LLM batch classification.
"""
import json
from solo_leveling.llm_client import call_llm
from solo_leveling.prompts import ATTRIBUTION_PROMPT
from solo_leveling.skill_library import SkillLibrary, _normalize_skill, _is_frozen


def _hash_body(body) -> str:
    if hasattr(body, "tolist"):
        body = body.tolist()
    return "".join(str(int(v)) for row in body for v in row)


def _build_skills_block(skills: list[dict]) -> str:
    """Compress each skill to (skill_id, l1_condition, top 2 positive leaves) as JSON."""
    out = []
    for s in skills:
        l2 = s.get("l2", {})
        sorted_pos = sorted(
            l2.get("positive", []),
            key=lambda l: -float(l.get("avg_gain", 0.0) or 0.0),
        )
        top_pos = [
            {"claim": l.get("claim", ""), "description": l.get("description", "")}
            for l in sorted_pos[:2]
        ]
        out.append({
            "skill_id": s.get("skill_id"),
            "l1_condition": s.get("l1", {}).get("condition", ""),
            "top_positive_leaves": top_pos,
        })
    return json.dumps(out, indent=2)


def _route_path_a(library: SkillLibrary, obs_dicts: list[dict], current_gen: int) -> int:
    """Trust-marker: write Path A obs into target skill.L3.

    Frozen (imported) targets are not written to — instead obs are routed to
    unassigned_pool with provenance fields, so NEW_INSIGHT can crystallize
    native skills that "inherit" the imported skill's pattern.
    """
    routed = 0
    skills = {s["skill_id"]: s for s in library.retrieve_all(include_fully_retired=True)}
    skills_to_save = set()

    for raw in obs_dicts:
        if raw.get("based_on_skill") is None:
            continue
        skill_id = raw["based_on_skill"]
        if skill_id not in skills:
            continue
        skill = skills[skill_id]

        # Frozen path-a routing: imported target -> pool with provenance
        if _is_frozen(skill):
            pool_obs = _to_pool_obs(library, raw, current_gen)
            pool_obs["source_origin"] = "frozen_path_a"
            pool_obs["source_imported_skill_id"] = skill_id
            pool_obs["source_leaf_id"] = raw.get("intended_leaf_id")
            library.append_to_pool(pool_obs)
            routed += 1
            continue

        # Native path: existing logic - resolve intended_leaf_id, append to L3.
        intended_leaf_id = raw.get("intended_leaf_id")
        if intended_leaf_id is not None:
            l2 = skill.get("l2", {})
            all_leaves = l2.get("positive", []) + l2.get("negative", [])
            known_leaf_ids = {l.get("leaf_id") for l in all_leaves}
            if intended_leaf_id not in known_leaf_ids:
                resolved = next(
                    (l.get("leaf_id") for l in all_leaves if l.get("claim") == intended_leaf_id),
                    None,
                )
                intended_leaf_id = resolved   # may be None
        body = raw["body"]
        if hasattr(body, "tolist"):
            body = body.tolist()
        new_obs = {
            "obs_id": library.get_next_obs_id(),
            "body": body,
            "fitness": float(raw["fitness"]),
            "parent_body_hash": raw.get("parent_body_hash"),
            "parent_fitness": raw.get("parent_fitness"),
            "gain": raw.get("gain"),
            "used_leaf_id": intended_leaf_id,
            "gen_observed": current_gen,
            "source": "path_a",
            "diagnosis_attempts": 0,
        }
        skill.setdefault("l3", {}).setdefault("observations", []).append(new_obs)
        skills_to_save.add(skill_id)
        routed += 1

    for sid in skills_to_save:
        library._save(skills[sid])
    return routed


def _attribute_path_g(library: SkillLibrary, obs_dicts: list[dict], current_gen: int, task_name: str) -> None:
    """LLM-batch classify Path G obs. Either route to a skill's L3 or write to pool."""
    path_g = [o for o in obs_dicts if o.get("based_on_skill") is None]
    if not path_g:
        return

    active_skills = library.retrieve(task_name, for_writing=True)

    # No skills to match against: dump all to pool
    if not active_skills:
        for raw in path_g:
            library.append_to_pool(_to_pool_obs(library, raw, current_gen))
        return

    skills_block = _build_skills_block(active_skills)
    designs_block = json.dumps(
        [{"local_index": i, "body": _body_as_list(raw["body"])} for i, raw in enumerate(path_g)],
        indent=2,
    )
    prompt = ATTRIBUTION_PROMPT.format(skills_block=skills_block, designs_block=designs_block)
    try:
        response = call_llm([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"[Attribution] Path G LLM call failed: {e}; routing all to pool")
        for raw in path_g:
            library.append_to_pool(_to_pool_obs(library, raw, current_gen))
        return

    by_index = {a["local_index"]: a for a in response.get("assignments", [])}
    skills_index = {s["skill_id"]: s for s in active_skills}
    skills_to_save = set()

    for i, raw in enumerate(path_g):
        a = by_index.get(i)
        skill_id = a.get("skill_id") if a else None
        if skill_id is None or skill_id not in skills_index:
            library.append_to_pool(_to_pool_obs(library, raw, current_gen))
            continue
        skill = skills_index[skill_id]
        new_obs = {
            "obs_id": library.get_next_obs_id(),
            "body": _body_as_list(raw["body"]),
            "fitness": float(raw["fitness"]),
            "parent_body_hash": raw.get("parent_body_hash"),
            "parent_fitness": raw.get("parent_fitness"),
            "gain": raw.get("gain"),
            "used_leaf_id": None,
            "gen_observed": current_gen,
            "source": "path_g",
            "diagnosis_attempts": 0,
        }
        skill.setdefault("l3", {}).setdefault("observations", []).append(new_obs)
        skills_to_save.add(skill_id)

    for sid in skills_to_save:
        library._save(skills_index[sid])


def _to_pool_obs(library: SkillLibrary, raw: dict, current_gen: int) -> dict:
    return {
        "obs_id": library.get_next_obs_id(),
        "body": _body_as_list(raw["body"]),
        "fitness": float(raw["fitness"]),
        "parent_body_hash": raw.get("parent_body_hash"),
        "parent_fitness": raw.get("parent_fitness"),
        "gain": raw.get("gain"),
        "gen_observed": current_gen,
    }


def _body_as_list(body) -> list:
    if hasattr(body, "tolist"):
        return body.tolist()
    return body


def re_attribute_pool(library: SkillLibrary, current_gen: int, task_name: str) -> int:
    """Re-attribute accumulated unassigned_pool obs against current (evolved) skill library.
    Returns number of obs successfully re-attributed (moved out of pool into a skill.L3).
    """
    pool = library.get_pool()
    if not pool:
        return 0

    active_skills = library.retrieve(task_name, for_writing=True)
    if not active_skills:
        return 0

    # Convert pool obs to raw_obs dicts so we can reuse _attribute_path_g logic
    raw_obs = [
        {
            "based_on_skill": None,
            "body": o["body"],
            "fitness": o.get("fitness"),
            "parent_body_hash": o.get("parent_body_hash"),
            "parent_fitness": o.get("parent_fitness"),
            "gain": o.get("gain"),
            "eval_status": "ok",
            "_pool_obs_id": o.get("obs_id"),  # remember original obs_id
            # Carry frozen-path-a provenance through re-attribution
            "_source_origin": o.get("source_origin"),
            "_source_imported_skill_id": o.get("source_imported_skill_id"),
            "_source_leaf_id": o.get("source_leaf_id"),
        }
        for o in pool
    ]

    skills_block = _build_skills_block(active_skills)
    designs_block = json.dumps(
        [{"local_index": i, "body": _body_as_list(raw["body"])} for i, raw in enumerate(raw_obs)],
        indent=2,
    )
    prompt = ATTRIBUTION_PROMPT.format(skills_block=skills_block, designs_block=designs_block)
    try:
        response = call_llm([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"[Attribution] re_attribute_pool LLM call failed: {e}; pool unchanged")
        return 0

    by_index = {a["local_index"]: a for a in response.get("assignments", [])}
    skills_index = {s["skill_id"]: s for s in active_skills}
    skills_to_save = set()
    matched_obs_ids = []

    for i, raw in enumerate(raw_obs):
        a = by_index.get(i)
        skill_id = a.get("skill_id") if a else None
        if skill_id is None or skill_id not in skills_index:
            continue
        skill = skills_index[skill_id]
        # Migrate from pool to this skill's L3 (preserve original obs_id)
        new_obs = {
            "obs_id": raw["_pool_obs_id"],
            "body": _body_as_list(raw["body"]),
            "fitness": float(raw["fitness"]) if raw["fitness"] is not None else None,
            "parent_body_hash": raw.get("parent_body_hash"),
            "parent_fitness": raw.get("parent_fitness"),
            "gain": raw.get("gain"),
            "used_leaf_id": None,
            "gen_observed": current_gen,  # mark re-attribution gen
            "source": "path_g",
            "diagnosis_attempts": 0,
            # Preserve frozen-path-a provenance
            "source_origin": raw.get("_source_origin"),
            "source_imported_skill_id": raw.get("_source_imported_skill_id"),
            "source_leaf_id": raw.get("_source_leaf_id"),
        }
        skill.setdefault("l3", {}).setdefault("observations", []).append(new_obs)
        skills_to_save.add(skill_id)
        matched_obs_ids.append(raw["_pool_obs_id"])

    if matched_obs_ids:
        library.remove_from_pool(matched_obs_ids)
    for sid in skills_to_save:
        library._save(skills_index[sid])
    return len(matched_obs_ids)


def run_attribution(library: SkillLibrary, obs_dicts: list[dict], current_gen: int, task_name: str) -> None:
    """Stage 1 (§4): dispatch obs to Path A or Path G handlers."""
    _route_path_a(library, obs_dicts, current_gen)
    _attribute_path_g(library, obs_dicts, current_gen, task_name)
