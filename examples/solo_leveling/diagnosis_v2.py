"""L2 Diagnosis stage (§6): rule-based update + LLM-driven leaf assignment.

Replaces the legacy split DIAGNOSE_POSITIVE / DIAGNOSE_NEGATIVE flow with a single
DIAGNOSE_LEAVES call per skill. Used only via v2 pipeline; v1.5 SkillLibrary.diagnose
is left intact for backward compat.
"""
import json
from solo_leveling.llm_client import call_llm
from solo_leveling.prompts import DIAGNOSE_LEAVES_PROMPT, _COLD_START_NOTE
from solo_leveling.skill_library import SkillLibrary, _is_frozen

DIAGNOSIS_ATTEMPTS_K = 3


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def run_diagnosis(
    library: SkillLibrary,
    current_gen: int,
    task_name: str,
    ablate_llm_diagnose: bool = False,
) -> None:
    """Stage 3 (§6): for each skill with new evidence this gen, run rule-based + LLM.

    When ablate_llm_diagnose=True, the _llm_diagnose_leaves step is skipped so
    L2 leaves never evolve via LLM. _rule_based_update still runs (it's idempotent
    on already-added obs_ids), and the idempotency guard last_l2_diagnosed_generation
    is intentionally left unadvanced so telemetry reflects 0 LLM diagnoses.
    """
    skills = {s["skill_id"]: s for s in library.retrieve_all(include_fully_retired=True)}
    for skill_id, skill in skills.items():
        if task_name not in skill.get("task_family", []):
            continue
        if _is_frozen(skill):
            continue  # imported skills never receive L2 updates
        if not _has_new_evidence(skill, current_gen):
            continue
        if skill.get("status", {}).get("last_l2_diagnosed_generation") == current_gen:
            continue   # idempotency guard
        _rule_based_update(skill, current_gen)
        if not ablate_llm_diagnose:
            _llm_diagnose_leaves(library, skill, current_gen)
        library._save(skill)


def _has_new_evidence(skill: dict, current_gen: int) -> bool:
    return any(
        o.get("gen_observed") == current_gen
        for o in skill.get("l3", {}).get("observations", [])
    )


def _incremental_mean(old: float, new_value: float, n_after: int) -> float:
    return old + (new_value - old) / n_after


def _rule_based_update(skill: dict, current_gen: int) -> None:
    """§6.2: incremental update for Path A obs already carrying used_leaf_id."""
    l2 = skill.get("l2", {})
    leaf_index = {l["leaf_id"]: l for l in l2.get("positive", []) + l2.get("negative", [])}
    observations = skill.get("l3", {}).get("observations", [])

    for obs in observations:
        if obs.get("source") != "path_a":
            continue
        if obs.get("gen_observed") != current_gen:
            continue
        leaf_id = obs.get("used_leaf_id")
        if not leaf_id:
            continue
        leaf = leaf_index.get(leaf_id)
        if leaf is None:
            continue
        oid = obs.get("obs_id")
        ids = leaf.setdefault("obs_ids", [])
        if oid in ids:
            continue
        ids.append(oid)
        gain = obs.get("gain")
        if gain is not None:
            leaf["avg_gain"] = _incremental_mean(
                float(leaf.get("avg_gain", 0.0)), float(gain), len(ids),
            )

    status = skill.setdefault("status", {})
    status["total_samples"] = len(observations)
    status["generation_updated"] = current_gen
    # §3.6 Invariant 5: avg_gain only over assigned obs
    assigned_gains = [
        float(o.get("gain"))
        for o in observations
        if o.get("used_leaf_id") is not None and o.get("gain") is not None
    ]
    status["avg_gain"] = sum(assigned_gains) / len(assigned_gains) if assigned_gains else 0.0
    # positive_rate: fraction of all gains > 0 (regardless of leaf assignment)
    all_gains = [
        float(o.get("gain"))
        for o in observations
        if o.get("gain") is not None
    ]
    n_pos = sum(1 for g in all_gains if g > 0)
    status["positive_rate"] = n_pos / len(all_gains) if all_gains else 0.0
    # last_l2_diagnosed_generation is NOT set here — only after LLM completes.


def _llm_diagnose_leaves(library: SkillLibrary, skill: dict, current_gen: int) -> None:
    l2 = skill.setdefault("l2", {})
    l2.setdefault("positive", [])
    l2.setdefault("negative", [])
    l2.setdefault("next_leaf_id_counter", 0)
    observations = skill.setdefault("l3", {}).setdefault("observations", [])

    unassigned = [
        o for o in observations
        if o.get("used_leaf_id") is None
        and int(o.get("diagnosis_attempts", 0)) < DIAGNOSIS_ATTEMPTS_K
    ]
    context = [
        o for o in observations
        if o.get("gen_observed") == current_gen and o.get("used_leaf_id") is not None
    ]

    l2_is_empty = not (l2["positive"] or l2["negative"])
    if not unassigned and not l2_is_empty:
        # Nothing to do — still advance idempotency guard
        skill.setdefault("status", {})["last_l2_diagnosed_generation"] = current_gen
        return

    leaves_payload = [
        {
            "leaf_id": l.get("leaf_id"),
            "claim": l.get("claim"),
            "description": l.get("description"),
            "polarity": _polarity_of(l, l2),
            "avg_gain": float(l.get("avg_gain", 0.0)),
        }
        for l in l2["positive"] + l2["negative"]
    ]
    unassigned_payload = [_obs_payload(o) for o in unassigned]
    context_payload = [_obs_payload(o) for o in context]
    gen_mean, gen_p25 = _gen_stats(observations, current_gen)

    cold_start_note = _COLD_START_NOTE if l2_is_empty else ""
    prompt = DIAGNOSE_LEAVES_PROMPT.format(
        l1_condition=skill.get("l1", {}).get("condition", "N/A"),
        leaves_json=json.dumps(leaves_payload, indent=2),
        unassigned_json=json.dumps(unassigned_payload, indent=2),
        context_json=json.dumps(context_payload, indent=2),
        gen_mean=gen_mean,
        gen_p25=gen_p25,
        cold_start_note=cold_start_note,
    )

    try:
        response = call_llm([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"[Diagnosis] DIAGNOSE_LEAVES LLM failed for {skill.get('skill_id')}: {e}")
        return  # do not advance idempotency guard; will retry next call

    # Increment diagnosis_attempts on every obs that was sent in
    sent_ids = {o.get("obs_id") for o in unassigned}
    for o in observations:
        if o.get("obs_id") in sent_ids:
            o["diagnosis_attempts"] = int(o.get("diagnosis_attempts", 0)) + 1

    # Standalone first; absorbed obs skip per-obs new_leaf
    standalone = response.get("standalone_new_leaves") or []
    absorbed = set()
    for s in standalone:
        leaf = _create_leaf(skill, s.get("polarity"), s.get("claim"), s.get("description"))
        leaf["obs_ids"] = [int(x) for x in s.get("supporting_obs_ids", [])]
        leaf["avg_gain"] = _mean_gain(observations, leaf["obs_ids"])
        _append_leaf(skill, leaf, s.get("polarity"))
        absorbed.update(leaf["obs_ids"])

    obs_index = {o.get("obs_id"): o for o in observations}
    leaf_index = {
        l.get("leaf_id"): (l, "positive" if l in l2["positive"] else "negative")
        for l in l2["positive"] + l2["negative"]
    }

    for a in (response.get("leaf_assignments") or []):
        oid = a.get("obs_id")
        decision = a.get("decision")
        obs = obs_index.get(oid)
        if obs is None:
            continue

        if decision == "new_leaf" and oid in absorbed:
            continue   # redundant; standalone already covered

        if decision == "match_existing":
            leaf_id = a.get("leaf_id")
            entry = leaf_index.get(leaf_id)
            if entry is None:
                continue
            leaf, _polarity = entry
            obs["used_leaf_id"] = leaf_id
            ids = leaf.setdefault("obs_ids", [])
            if oid not in ids:
                ids.append(oid)
                gain = obs.get("gain")
                if gain is not None:
                    leaf["avg_gain"] = _incremental_mean(
                        float(leaf.get("avg_gain", 0.0)), float(gain), len(ids),
                    )
            update = a.get("description_update") or {}
            mode = update.get("mode")
            text = (update.get("text") or "").strip()
            from solo_leveling.config import MAX_LEAF_DESCRIPTION_WORDS
            if mode == "overwrite" and text:
                leaf["description"] = _cap_words(text, MAX_LEAF_DESCRIPTION_WORDS)
            elif mode == "append" and text:
                candidate = (leaf.get("description", "") + " " + text).strip()
                if len(candidate.split()) > MAX_LEAF_DESCRIPTION_WORDS:
                    # Auto-convert to overwrite to keep latest signal
                    leaf["description"] = _cap_words(text, MAX_LEAF_DESCRIPTION_WORDS)
                else:
                    leaf["description"] = candidate
            # mode == None or text empty → no change

        elif decision == "new_leaf":
            leaf = _create_leaf(skill, a.get("polarity"), a.get("claim"), a.get("description"))
            leaf["obs_ids"] = [oid]
            leaf["avg_gain"] = float(obs.get("gain") or 0.0)
            _append_leaf(skill, leaf, a.get("polarity"))
            obs["used_leaf_id"] = leaf["leaf_id"]

        # decision == "no_leaf": leave used_leaf_id as None (default)

    # Recompute status.avg_gain after LLM-driven assignment changes (Invariant 5)
    assigned_gains = [
        float(o.get("gain"))
        for o in observations
        if o.get("used_leaf_id") is not None and o.get("gain") is not None
    ]
    status = skill.setdefault("status", {})
    status["avg_gain"] = sum(assigned_gains) / len(assigned_gains) if assigned_gains else 0.0
    # positive_rate: fraction of all gains > 0 (regardless of leaf assignment)
    all_gains = [
        float(o.get("gain"))
        for o in observations
        if o.get("gain") is not None
    ]
    n_pos = sum(1 for g in all_gains if g > 0)
    status["positive_rate"] = n_pos / len(all_gains) if all_gains else 0.0
    # Idempotency guard set last (crash/resume safe)
    status["last_l2_diagnosed_generation"] = current_gen


def _obs_payload(obs: dict) -> dict:
    return {
        "obs_id": obs.get("obs_id"),
        "body": obs.get("body"),
        "gain": obs.get("gain"),
        "source": obs.get("source"),
        "used_leaf_id": obs.get("used_leaf_id"),
    }


def _gen_stats(observations: list[dict], current_gen: int) -> tuple[float, float]:
    cur = [
        float(o.get("gain"))
        for o in observations
        if o.get("gen_observed") == current_gen and o.get("gain") is not None
    ]
    if not cur:
        return 0.0, 0.0
    cur_sorted = sorted(cur)
    mean = sum(cur_sorted) / len(cur_sorted)
    idx = max(0, len(cur_sorted) // 4 - 1)
    p25 = cur_sorted[idx]
    return mean, p25


def _create_leaf(skill: dict, polarity: str, claim: str, description: str) -> dict:
    l2 = skill["l2"]
    counter = int(l2.get("next_leaf_id_counter", 0) or 0)
    prefix = "pos" if polarity == "positive" else "neg"
    leaf_id = f"{prefix}_{counter}"
    l2["next_leaf_id_counter"] = counter + 1
    return {
        "leaf_id": leaf_id,
        "claim": claim or "",
        "description": description or "",
        "obs_ids": [],
        "birth_obs_count": 0,
        "avg_gain": 0.0,
    }


def _append_leaf(skill: dict, leaf: dict, polarity: str) -> None:
    bucket = skill["l2"]["positive"] if polarity == "positive" else skill["l2"]["negative"]
    bucket.append(leaf)


def _polarity_of(leaf: dict, l2: dict) -> str:
    return "positive" if leaf in l2.get("positive", []) else "negative"


def _mean_gain(observations: list[dict], obs_ids: list[int]) -> float:
    obs_index = {o.get("obs_id"): o for o in observations}
    gains = []
    for oid in obs_ids:
        o = obs_index.get(oid)
        if o is not None and o.get("gain") is not None:
            gains.append(float(o["gain"]))
    return sum(gains) / len(gains) if gains else 0.0
