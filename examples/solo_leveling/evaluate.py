import copy
from argparse import Namespace
from pathlib import Path

import numpy as np

from ppo.run import run_ppo
import utils.mp_group as mp


def evaluate_batch(
    designs: list[dict],
    env_name: str,
    save_dir: str,
    generation: int,
    num_cores: int,
    ppo_args: Namespace,
) -> list[dict]:
    """
    Evaluate a batch of designs in parallel using run_ppo.

    Each design dict must have:
      - "body": np.ndarray (5x5)
      - "connections": np.ndarray
      - "label": int

    Returns the same list enriched with a "fitness" key.
    """
    save_root = Path(save_dir)
    controller_dir = save_root / f"generation_{generation}" / "controller"
    structure_dir = save_root / f"generation_{generation}" / "structure"
    controller_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)

    # Pre-save structure files (matches GA log format)
    for d in designs:
        label = d["label"]
        np.savez(
            structure_dir / f"{label}.npz",
            d["body"],
            d["connections"],
        )

    # Build parallel job group
    group = mp.Group()
    fitnesses = [0.0] * len(designs)

    for idx, d in enumerate(designs):
        label = d["label"]
        model_save_dir = str(controller_dir)
        model_save_name = str(label)

        def make_callback(i):
            def callback(fitness):
                fitnesses[i] = fitness
            return callback

        ppo_args_copy = copy.copy(ppo_args)
        group.add_job(
            run_ppo,
            (ppo_args_copy, d["body"], env_name, model_save_dir, model_save_name, d["connections"]),
            make_callback(idx),
        )

    group.run_jobs(num_cores)

    # Enrich designs with fitness
    results = []
    for idx, d in enumerate(designs):
        enriched = dict(d)
        enriched["fitness"] = fitnesses[idx]
        results.append(enriched)

    return results
