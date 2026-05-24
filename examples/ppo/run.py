import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import evogym.envs  # noqa: F401 — registers custom environments with gymnasium

from ppo.eval import eval_policy
from ppo.callback import EvalCallback

def _linear_schedule(initial_value: float):
    """Linear learning-rate schedule: matches paper Table 6 `use linear lr decay: True`."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def run_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """
    Run ppo and return the best reward achieved during evaluation.
    """

    # Link PPO seed to caller-provided seed when available (e.g. GA --seed),
    # so that the entire pipeline varies with one knob. Falls back to the
    # passed-in `seed` arg for scripts that don't expose --seed.
    effective_seed = getattr(args, 'seed', None)
    if effective_seed is None:
        effective_seed = seed

    # Parallel environments (paper: num_processes=4)
    vec_env = make_vec_env(env_name, n_envs=args.n_envs, seed=effective_seed, env_kwargs={
        'body': body,
        'connections': connections,
    })

    # Eval Callback
    callback = EvalCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        verbose=args.verbose_ppo,
    )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=args.verbose_ppo,
        learning_rate=_linear_schedule(args.learning_rate),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        device="cpu"
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval
    )
    
    return callback.best_reward