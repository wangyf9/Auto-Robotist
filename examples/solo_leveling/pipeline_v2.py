"""Top-level v2 pipeline orchestrator (§2 流水线总览).

Wires Pre-filter → Attribution → L1 New Insight → L2 Diagnosis for one generation.
"""
from solo_leveling.skill_library import SkillLibrary
from solo_leveling.prefilter import filter_evaluated_obs
from solo_leveling.attribution import run_attribution, re_attribute_pool
from solo_leveling.new_insight_v2 import run_new_insight
from solo_leveling.diagnosis_v2 import run_diagnosis
from solo_leveling.config import RE_ATTRIBUTION_POOL_THRESHOLD


def run_one_generation(
    library: SkillLibrary,
    raw_obs: list[dict],
    current_gen: int,
    task_name: str,
    ablate_llm_diagnose: bool = False,
) -> dict:
    """Execute one generation's post-evaluation pipeline.

    Args:
        library: SkillLibrary (mutated in place via _save calls).
        raw_obs: dicts from PPO evaluator. Each must contain at minimum:
            body, fitness, parent_body_hash, parent_fitness, gain, eval_status.
            Path A obs additionally have based_on_skill and intended_leaf_id.
        current_gen: integer generation number.
        task_name: env name (e.g., "Walker-v0").

    Returns:
        Summary dict with dropped_count, kept_count, new_skill, diagnosed_skill_count.
    """
    # Re-attribute accumulated pool obs against evolved skill library (threshold-gated)
    pool_size_before = len(library.get_pool())
    if pool_size_before >= RE_ATTRIBUTION_POOL_THRESHOLD:
        n_re_attributed = re_attribute_pool(library, current_gen, task_name)
        if n_re_attributed > 0:
            print(f"  [v2 pipeline] re-attributed {n_re_attributed} pool obs to skills (was pool={pool_size_before})")

    kept, dropped = filter_evaluated_obs(raw_obs)
    run_attribution(library, kept, current_gen, task_name)

    # Build current-gen obs list across all retrieved skills (for fallback)
    current_gen_obs = []
    for s in library.retrieve(task_name, for_writing=True):
        for o in s.get("l3", {}).get("observations", []):
            if o.get("gen_observed") == current_gen:
                current_gen_obs.append(o)

    new_skill = run_new_insight(library, current_gen, current_gen_obs, task_name)
    run_diagnosis(library, current_gen, task_name, ablate_llm_diagnose=ablate_llm_diagnose)

    diagnosed = sum(
        1 for s in library.retrieve(task_name, for_writing=True)
        if s.get("status", {}).get("last_l2_diagnosed_generation") == current_gen
    )
    return {
        "dropped_count": len(dropped),
        "kept_count": len(kept),
        "new_skill": new_skill["skill_id"] if new_skill else None,
        "diagnosed_skill_count": diagnosed,
        "pool_size": len(library.get_pool()),
    }
