"""Pipeline pre-filter (§1.5): drop suspicious / failed eval obs before pipeline."""
from typing import Tuple


def filter_evaluated_obs(obs_list: list[dict]) -> Tuple[list[dict], list[dict]]:
    """Split obs by eval_status. Returns (kept, dropped).

    Only obs with eval_status == "ok" are kept. crash / nan / timeout / suspicious
    are excluded from all downstream stages.
    """
    kept = [o for o in obs_list if o.get("eval_status") == "ok"]
    dropped = [o for o in obs_list if o.get("eval_status") != "ok"]
    return kept, dropped
