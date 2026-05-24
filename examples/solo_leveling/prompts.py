VOXEL_LEGEND = """
Voxel type integers (must match EvoGym VOXEL_TYPES exactly):
  0 = EMPTY   (no voxel)
  1 = RIGID   (structural, cannot actuate)
  2 = SOFT    (deformable, passive)
  3 = H_ACT   (horizontal actuator — expands/contracts horizontally)
  4 = V_ACT   (vertical actuator — expands/contracts vertically)
  5 = FIXED   (anchored to world, cannot move)
"""

_TC_SAME_GRID_BASE = """=== Transfer Context ===
The skill library below comes from a prior run on the same task
({current_env}, {source_grid} grid, source experiment "{source_exp}").
Treat the L1/L2 rules as validated patterns from previous experimentation.
Note: avg_gain values reflect the prior run's parent fitness distribution;
treat them as relative ranking signals, not absolute predictions.
"""

_TC_CROSS_GRID_BASE = """=== Transfer Context ===
Current task: {current_env} ({current_grid} voxel grid).
The skill library below comes from a prior run on a related task with a
{source_grid} voxel grid (source experiment "{source_exp}").

The skills' L1/L2 rules describe abstract structural principles (e.g.
"vertical rails joined by horizontal crossbeam") that should generalize
across grid sizes. The current {current_grid} grid affords richer / more
redundant structures than {source_grid}.
"""

_TC_SAME_GRID_ELITE_ADDENDUM = """
Reference designs (top-fitness exemplars from the source run) are also
provided below. Use them as concrete examples of what the L1/L2 rules
look like in practice.
"""

_TC_CROSS_GRID_ELITE_ADDENDUM = """
Reference designs (top-fitness exemplars) are also provided below as
concrete {source_grid} examples — extract structural patterns from them,
do NOT copy voxel-level arrangements directly.
"""

# Shared body schema line used in all PROPOSE prompt JSON examples
_BODY_ROW = "[int, int, int, int, int]"
BODY_SCHEMA = f"[{_BODY_ROW}, {_BODY_ROW}, {_BODY_ROW}, {_BODY_ROW}, {_BODY_ROW}]"

# Shared structural requirements for all PROPOSE prompts
BODY_REQUIREMENTS = """- body is a 5x5 grid of integers (rows top-to-bottom, columns left-to-right)
- Each integer must be 0, 1, 2, 3, 4, or 5
- The robot MUST be fully connected (all non-empty voxels reachable via 4-connectivity, no isolated groups)
- MUST contain at least one actuator (3=H_ACT or 4=V_ACT)"""

PROPOSE_INIT_PROMPT = """You are an expert soft-robot morphology designer for the EvoGym simulation platform.

Task: {task_desc}

Voxel grid legend:
{voxel_legend}

Generate exactly {n_designs} diverse robot designs as a JSON object with this schema:
{{
  "designs": [
    {{
      "body": """ + BODY_SCHEMA + """,
      "reasoning": "brief explanation of design choices"
    }}
  ]
}}

Requirements:
""" + BODY_REQUIREMENTS + """
- Explore diverse structures — vary leg count, body shape, actuator placement, symmetry
- Every design must be structurally different from each other
"""

PROPOSE_MUTATE_PROMPT = """You are an expert soft-robot morphology designer for the EvoGym simulation platform.

Task: {task_desc}

{transfer_context_block}
Voxel grid legend:
{voxel_legend}

Here is the parent design to mutate (fitness={parent_fitness:.3f}):
{parent_body}

Skill assignments for this proposal batch:
{skill_assignments_block}

{static_reference_block}
Previous mutation history on this exact parent:
{history_block}

Your task:
Propose exactly {n_designs} new mutations of this parent, one for each assigned slot above,
using a two-part process:

1. Direction from the assigned skill:
   Use the skill's L1 condition as the target structural archetype for that slot.
   If the skill has no L2 rules yet, still move the parent toward the L1 condition.

2. Tactics from L2 rules and exact-parent history:
   Use L2 positive rules as helpful sub-patterns when they fit this parent.
   Avoid L2 negative rules when relevant.
   Use exact-parent history to avoid repeats and avoid edits that already failed on this parent.

Each history entry gives you raw evidence only:
- the exact child_fitness achieved on this parent
- the voxel_diff from parent to child
- the full child_body after that mutation

Use this evidence directly to judge which edits seem promising, harmful, or ambiguous.

Hard constraints:
- You must produce new child bodies distinct from all history entries listed above.
- For each slot, use the specific assigned skill for that slot only.
- For each slot, also output `intended_leaf_id`: the L2 leaf you are trying to instantiate.
  IMPORTANT: this MUST be the leaf_id (e.g. "pos_0", "neg_1") shown in the skill's L2 rules block, NOT the claim string. Look for "leaf_id=..." in the rules listing.
  If the assigned skill has no leaves yet, output null.
- If an assigned skill is not a natural fit for this parent, propose the smallest edit that still moves in that skill direction without destroying the parent's core structure.
- Do not refuse to propose. An imperfect mutation is still useful learning material.

Generate exactly {n_designs} mutated variations of this parent as a JSON object with this schema:
{{
  "designs": [
    {{
      "slot_index": 0,
      "body": """ + BODY_SCHEMA + """,
      "reasoning": "what you changed from the parent and why",
      "based_on_skill": "skill_id string or null",
      "intended_leaf_id": "leaf_id string or null"
    }}
  ]
}}

Requirements:
""" + BODY_REQUIREMENTS + """
- Each design should modify {mutation_range} voxels from the parent — keep what works, change what could improve
- L1 chooses the mutation direction; L2 rules and exact-parent history choose the concrete edits
- History can veto repeated or clearly harmful edits, but it should not replace the assigned L1 direction
- Do not repeat any exact child body already present in the history block
- Return exactly one design per slot_index listed above
- Every mutation must be structurally different from the parent AND from each other
"""

UNRETRIEVED_DIAGNOSE_PROMPT = """You are analyzing whether successful robot designs from one task match patterns described by skills learned from other tasks.

Current task: {current_task}

Top-performing designs from this generation:
{top_designs}

Skills from other tasks (NOT currently applied to {current_task}):
{non_retrieved_skills}

For each skill, determine if ANY of the top designs clearly exhibit the structural pattern
described by that skill's L1 condition and L2 positive rules.

Return JSON:
{{
  "matches": [
    {{
      "skill_id": "the matching skill's id",
      "reason": "brief explanation of why the top design matches this skill's pattern"
    }}
  ]
}}

Only include genuine structural matches. Return empty matches list if no matches found.
"""

DIAGNOSE_POSITIVE_PROMPT = """You are logging structural evidence for a robot design skill.

Skill:
  L1 condition: {l1_condition}

Current L2 positive rules for this skill:
{l2_positive}

Existing positive observations for this skill (facts only):
{existing_positives}

New designs this generation that used this skill and scored ABOVE gen_mean:
{new_designs}

For each new design above, provide one compact L2 positive rule if this design
reveals a reusable sub-pattern under the L1 archetype. The rule may repeat an
existing rule only when the evidence truly supports the same sub-pattern.

Constraints:
- Use structural category language (e.g., "symmetric actuator", "low center of mass",
  "wide support base"). DO NOT use voxel coordinates, row/col indices, or exact voxel counts.
- L2 rule claim must be 1-4 words in snake_case.
- L2 rule description should be one concise structural sentence.
- If no reusable sub-pattern is present, return null for that design's rule.

Return JSON:
{{
  "observations": [
    {{
      "body_index": 0,
      "rule": {{"claim": "short_snake_case", "description": "..."}} or null
    }},
    {{
      "body_index": 1,
      "rule": {{"claim": "short_snake_case", "description": "..."}} or null
    }}
  ]
}}
"""

DIAGNOSE_NEGATIVE_PROMPT = """You are logging structural evidence for a robot design skill.

Skill:
  L1 condition: {l1_condition}

Current L2 avoid rules for this skill:
{l2_negative}

Existing negative observations for this skill (facts only):
{existing_negatives}

New designs this generation that used this skill and scored BELOW p25:
{new_designs}

For each new design above, provide one compact L2 avoid rule if this design
reveals a reusable failure sub-pattern under the L1 archetype.

Constraints:
- Use structural category language (e.g., "top-heavy mass", "disconnected cluster",
  "missing anchor"). DO NOT use voxel coordinates, row/col indices, or exact voxel counts.
- L2 rule claim must be 1-4 words in snake_case.
- L2 rule description should be one concise structural sentence.
- If no reusable failure sub-pattern is present, return null for that design's rule.

Return JSON:
{{
  "observations": [
    {{
      "body_index": 0,
      "rule": {{"claim": "short_snake_case", "description": "..."}} or null
    }},
    {{
      "body_index": 1,
      "rule": {{"claim": "short_snake_case", "description": "..."}} or null
    }}
  ]
}}
"""

CONSOLIDATE_CLUSTER_PROMPT = """You are checking a robot design skill library for obvious duplicate L1 skill identities.

Skills in the library (L1 identity only):
{skills_full_content}

Return merge clusters only for obvious duplicate or near-duplicate L1 skill identities.
The main comparison target is L1 condition: merge skills only when they describe the
same main structural archetype in different words.

L1 structure is only a weak hint. Two skills may both use "frame", "rail", or "column"
and still be different if their L1 conditions describe different connected layouts.

Return JSON:
{{
  "clusters": [
    {{"group_label": "short_name (max 3 words)", "skill_ids": ["id1", "id2"], "reason": "why these are the same mechanism"}}
  ]
}}

Rules:
- group_label must be max 3 words, descriptive, snake_case
- Return only multi-skill clusters that should be merged; return [] if no obvious duplicates exist
- Do not create single-skill groups
- Do not merge skills just because they share the same performance goal, same task, or same broad structure word
- If there is any meaningful doubt, keep the skills separate
"""

NEW_INSIGHT_PROMPT = """You are a robot design analyst. Your job is to add useful L1 structural skill identities for robot morphology search.

High-fitness designs (top 6 from this generation, body grids + fitness):
{high_designs}

Low-fitness designs (bottom 6 from this generation, body grids + fitness):
{low_designs}

Existing skills for this task (each with L1 condition + top 2 positive L2 leaves):
{existing_skills_block}

Look for at most ONE new L1 structural archetype in this generation.
Start from the high-fitness designs: if several of them share a simple main structure,
you may add it as a new skill. Use the low-fitness designs only as a light contrast
signal: it is enough if the structure is absent, broken, weaker, or less coherent there.
Be willing to add a new L1 when the high designs show a reusable structure that is not
an obvious duplicate of an existing L1 condition. Return no_add only when there is no
clear nameable structure in the high designs, or the best candidate is almost the same
as an existing L1 condition.

L1 should describe the robot's main connected load-bearing arrangement. Use simple
structure words such as frame, rail, column, bridge, arch, tripod, fork, wedge, tail,
crawler, shell, or beam. Avoid both performance goals and exact voxel-level details.
New Insight only creates the L1 identity; L2 and L3 must stay empty at birth.

Return JSON:
{{
  "decision": {{
    "action": "add" or "no_add",
    "inspired_obs_ids": [<int>, ...],
    "skill": {{
      "skill_id": "short_name (max 3 words, snake_case, no version suffix)",
      "task_family": ["{task_name}"],
      "condition": "same text as l1.condition",
      "l1": {{
        "structure": "one coarse structure word",
        "condition": "10-25 words describing one concrete structural archetype"
      }},
      "l2": {{
        "positive": [],
        "negative": [],
        "next_leaf_id_counter": 0
      }},
      "l3": {{
        "observations": [],
        "next_obs_id_counter": 0
      }}
    }} or null,
    "reasoning": {{
      "supporting_high_labels": [<int>, ...],
      "contrast_signal": "short explanation of the high-vs-low structural difference",
      "nearest_existing_skill": "skill_id string or null",
      "duplicate_risk": "none" or "low" or "high",
      "why_add_or_no_add": "short explanation"
    }}
  }}
}}

Rules:
- This is a lightweight discovery step, not a strict filter.
- Add when the high designs reveal a simple reusable L1 structure and duplicate_risk is not high.
- Reject only generation-level summaries, performance goals, local patches, voxel-coordinate descriptions, and near-duplicate L1 conditions.
- skill_id must be max 3 words in snake_case, with no version suffix.
- If action is "add", inspired_by_label must be the Label of the design that best exemplifies this pattern; if no_add, set skill and inspired_by_label to null.
- For any added skill, l2.positive, l2.negative, and l3.observations MUST be empty lists.
- reasoning is audit-only. It will not be shown to later stages, so be explicit and honest.
- Candidates have gain > 0 (computed as child_fitness - parent_fitness). Only consider these for new L1 archetypes.
- When evaluating duplicate_risk, compare against both L1 conditions AND top_positive_leaves of existing skills.
  Two skills with similar archetype but different positive sub-patterns may still be distinct.
- inspired_obs_ids must list the obs_ids from the input that best exemplify this pattern (use the obs_id values shown).
{low_skill_hint}"""


ATTRIBUTION_PROMPT = """You are classifying robot designs by their main structural archetype.

Skill library (each skill is described by its L1 archetype plus top positive sub-patterns):
{skills_block}

Designs to classify:
{designs_block}

For each design, determine which skill it structurally matches.

Use L1 condition as the PRIMARY criterion: does the design exhibit the main
load-bearing arrangement described by L1? Use top_positive_leaves as supplementary
evidence to recognize concrete instances of successful sub-patterns within
that archetype.

If the design does not clearly match any skill's L1 archetype, return null.

Constraints:
- Use structural matching only — do not consider fitness or task performance.
- A design may match at most one skill (return the best match).
- Prefer null over a weak match. Skills that don't fit will not gain useful evidence.
- L1 dominates over L2: if a design clearly matches one skill's L1 but somewhat
  resembles another skill's positive leaves, attribute by L1.

Return JSON:
{{
  "assignments": [
    {{"local_index": 0, "skill_id": "skill_id" or null, "reason": "short structural reason"}}
  ]
}}
"""


DIAGNOSE_LEAVES_PROMPT = """You are diagnosing L2 leaves for a robot design skill.

Skill L1 condition: {l1_condition}

Existing L2 leaves (with current statistics):
{leaves_json}

Unassigned observations (need leaf decisions):
{unassigned_json}

Context observations (already assigned this generation; for distribution awareness only):
{context_json}

Generation statistics: gen_mean={gen_mean:.3f}, p25={gen_p25:.3f} (context only, not primary criterion).

{cold_start_note}

For each unassigned obs, decide its leaf assignment. Optionally propose new standalone leaves
that capture cross-cutting patterns visible across multiple obs.

Decision rules:
- Primary criterion: gain (= child_fitness - parent_fitness). gen_mean / p25 are weak context.
- Positive leaf creation (lenient): obs.gain > 0 AND reusable structural sub-pattern.
- Negative leaf creation (strict): obs.gain << 0 (significantly negative) AND obvious failure structure.
- Prefer "no_leaf" when no clear sub-pattern; the obs will be re-evaluated in future generations
  (up to a hard cap of 3 attempts).
- standalone_new_leaves: only when >= 2 obs share the same not-yet-captured sub-pattern.
- description_update applies only to "match_existing" decisions.

Refinement granularity rules (apply to all claim/description text including standalone leaves):
- ALLOWED: relative regions ("lower-right corner", "upper half", "leftmost column"),
  shape language ("U-shape opening upward", "tapered top"),
  counts and proportions ("4 H_ACT", "30% RIGID", "at least 2 anchors"),
  structural relations ("anchor connects to lower rail").
- FORBIDDEN: exact voxel coordinates ("voxel at (5,4)"), numeric row/column indices
  ("row 0", "column 4"), full-body matrix templates.
- Prefer "approximately N" / "at least N" over rigid "exactly N".

Return JSON:
{{
  "leaf_assignments": [
    {{"obs_id": <int>, "decision": "match_existing", "leaf_id": "<id>",
      "description_update": {{"mode": "overwrite" | "append" | null, "text": "..." or null}}}},
    {{"obs_id": <int>, "decision": "new_leaf", "polarity": "positive" | "negative",
      "claim": "snake_case_1_to_4_words", "description": "structural sentence"}},
    {{"obs_id": <int>, "decision": "no_leaf"}}
  ],
  "standalone_new_leaves": [
    {{"polarity": "positive" | "negative", "claim": "...", "description": "...",
      "supporting_obs_ids": [<int>, <int>, ...]}}
  ]
}}
"""


_COLD_START_NOTE = (
    "** This skill currently has NO leaves (cold-start). Treat this call as initial leaf "
    "bootstrap: every unassigned obs that exhibits reusable structure should be considered "
    "for new_leaf. match_existing is impossible (no leaves to match against). **"
)

