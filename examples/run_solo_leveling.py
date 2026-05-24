"""
Solo Leveling Agent — entry point.

Run from examples/ directory:
    python run_solo_leveling.py --exp-name solo_walker_001 --env-name Walker-v0 --num-cores 12 --total-gen 20

Smoke test (fast):
    python run_solo_leveling.py --exp-name smoke_test --total-gen 2 --num-cores 4
"""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ensure examples/ directory is on sys.path so relative imports work
sys.path.insert(0, str(Path(__file__).parent))

from argparse import Namespace

import solo_leveling.config as cfg
from solo_leveling.agent import SoloLevelingAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Solo Leveling LLM Agent for EvoGym")
    parser.add_argument("--exp-name", type=str, default="solo_exp_001",
                        help="Experiment name (used for save directory)")
    parser.add_argument("--env-name", type=str, default=cfg.ENV_NAME,
                        help="EvoGym environment name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global seed for Python/numpy RNG and PPO env seeding")
    parser.add_argument("--num-cores", type=int, default=cfg.NUM_CORES,
                        help="Number of parallel PPO workers")
    parser.add_argument("--total-gen", type=int, default=cfg.TOTAL_GENERATIONS,
                        help="Total number of generations to run")
    parser.add_argument("--n-designs", type=int, default=cfg.N_DESIGNS_PER_GEN,
                        help="Designs proposed per generation")
    parser.add_argument("--save-root", type=str, default=str(Path(__file__).parent / "saved_data" / "solo_leveling"),
                        help="Root directory for saving results")
    parser.add_argument("--total-timesteps", type=int, default=cfg.PPO_ARGS.total_timesteps,
                        help="PPO timesteps per design evaluation")
    parser.add_argument("--max-evaluations", type=int, default=cfg.MAX_EVALUATIONS,
                        help="Hard cap on total PPO evaluations (0 = unlimited)")
    parser.add_argument("--structure-shape", "--structure_shape", dest="structure_shape",
                        nargs=2, type=int, default=cfg.STRUCTURE_SHAPE, metavar=("HEIGHT", "WIDTH"),
                        help=f"Robot grid shape as HEIGHT WIDTH (default: {cfg.STRUCTURE_SHAPE[0]} {cfg.STRUCTURE_SHAPE[1]})")
    parser.add_argument("--init-skill-library", type=str, default=None,
        help="Path to source experiment's skills/ directory. Triggers Payload A+B+C "
             "import. The source experiment ROOT is the parent of this path; that's "
             "where elite extraction looks for state_snapshot.json.")
    parser.add_argument("--retrieve-as-task", type=str, default=None,
        help="Append this task to imported skills' task_family at copy time. "
             "Use for cross-grid transfer (e.g. --retrieve-as-task Walker-v0-10x10). "
             "Source library on disk is NOT modified.")
    parser.add_argument("--static-active-gens", type=int, default=5,
        help="Number of generations the static prior stays active in propose-time "
             "retrieve. After this, propose switches to native-only retrieval. "
             "Ignored when --init-skill-library is not provided.")
    parser.add_argument("--n-static-elites", type=int, default=5,
        help="Number of source elite designs to extract as reference exemplars. "
             "Ignored when --no-elite-inject is set or --init-skill-library missing.")
    parser.add_argument("--elite-inject", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to inject source elite designs as reference exemplars. "
             "Default: True. Use --no-elite-inject for ablation.")
    parser.add_argument("--init-designs-path", type=str, default=None,
                        help="Path to a directory containing pre-evaluated Gen 0 designs "
                             "(structure/*.npz + output.txt). If given, Gen 0 skips LLM "
                             "propose and PPO eval, loading body+fitness directly.")
    parser.add_argument("--mutation-range", type=str, default=cfg.MUTATION_RANGE,
                        help=f"LLM mutation voxel-count range as 'MIN-MAX'. "
                             f"E.g. '1-3' for 5x5, '1-10' for 10x10 (lower bound 1 keeps "
                             f"fine-grained refinement after convergence; upper bound ~10%% "
                             f"of voxel count for aggressive exploration). "
                             f"Default: {cfg.MUTATION_RANGE}")
    parser.add_argument("--ablate-diagnose", action="store_true",
        help="Ablation: skip _llm_diagnose_leaves so L2 stops evolving (strong-L2 removal). "
             "Keeps attribution/L3 accumulation/rule-based stats/add/merge intact.")
    parser.add_argument("--ablate-merge", action="store_true",
        help="Ablation: skip SkillLibrary.consolidate (no skill merging across the run).")
    parser.add_argument("--pure-llm", action="store_true",
        help="Ablation: set N_LLM_SLOTS=N_DESIGNS_PER_GEN, N_GA_SLOTS=0, disable LLM->GA overflow. "
             "Generation is pure Path A; size may be < N_DESIGNS_PER_GEN if LLM can't fill all slots.")
    parser.add_argument("--ablate-l2-injection", action="store_true",
        help="Ablation: in PROPOSE prompt's per-slot block, show only L1 condition; "
             "L2 positive/avoid rule lines are omitted (weak-L2 removal). L2 still grows in library.")
    return parser.parse_args()


def build_config(args) -> Namespace:
    """Merge CLI args with defaults from config.py into a single Namespace."""
    config = Namespace(
        # Agent
        ENV_NAME=args.env_name,
        STRUCTURE_SHAPE=tuple(args.structure_shape),
        N_DESIGNS_PER_GEN=args.n_designs,
        N_ELITE_PARENTS=cfg.N_ELITE_PARENTS,
        ELITE_SIZE=cfg.ELITE_SIZE,
        N_LLM_SLOTS=cfg.N_LLM_SLOTS,
        N_GA_SLOTS=cfg.N_GA_SLOTS,
        MUTATION_RANGE=args.mutation_range,
        NUM_CORES=args.num_cores,
        TOTAL_GENERATIONS=args.total_gen,
        MAX_EVALUATIONS=args.max_evaluations,
        # Ablation flags
        ABLATE_DIAGNOSE=args.ablate_diagnose,
        ABLATE_MERGE=args.ablate_merge,
        ABLATE_GA=args.pure_llm,
        ABLATE_L2_INJECTION=args.ablate_l2_injection,
        # PPO
        PPO_ARGS=copy.copy(cfg.PPO_ARGS),
    )
    if args.pure_llm:
        config.N_LLM_SLOTS = config.N_DESIGNS_PER_GEN
        config.N_GA_SLOTS = 0
    config.PPO_ARGS.total_timesteps = args.total_timesteps
    config.PPO_ARGS.seed = args.seed
    return config


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)

    print(f"Starting Solo Leveling Agent")
    print(f"  Experiment : {args.exp_name}")
    print(f"  Environment: {args.env_name}")
    print(f"  Grid shape : {config.STRUCTURE_SHAPE}")
    print(f"  Seed       : {args.seed}")
    print(f"  Generations: {config.TOTAL_GENERATIONS}")
    print(f"  Designs/gen: {config.N_DESIGNS_PER_GEN}")
    print(f"  Cores      : {config.NUM_CORES}")
    print(f"  Save root  : {args.save_root}")
    print(f"  Ablations  : diagnose={args.ablate_diagnose} merge={args.ablate_merge} "
          f"pure_llm={args.pure_llm} l2_inject={args.ablate_l2_injection}")

    agent = SoloLevelingAgent(
        env_name=args.env_name,
        exp_name=args.exp_name,
        save_root=args.save_root,
        config=config,
        init_skill_library=args.init_skill_library,
        init_designs_path=args.init_designs_path,
        retrieve_as_task=args.retrieve_as_task,
        elite_inject=args.elite_inject,
        n_static_elites=args.n_static_elites,
        static_active_gens=args.static_active_gens,
    )

    summary = agent.run()

    print(f"\nDone! {len(summary)} generations completed.")
    if summary:
        best_vals = [s["best_fitness"] for s in summary if s.get("best_fitness") is not None]
        if best_vals:
            print(f"Best fitness overall: {max(best_vals):.3f}")
        else:
            print("No completed evaluations (all generations skipped).")
    print(f"Results saved to: {args.save_root}/logs/{args.exp_name}/")


if __name__ == "__main__":
    main()
