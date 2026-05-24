from argparse import Namespace

# PPO hyperparameters — aligned with EvoGym paper (Table 6) and baseline_upstepper script.
# train iters 1000 × num_steps 128 × num_processes 4 = 512_000 total timesteps.
PPO_ARGS = Namespace(
    total_timesteps=512_000,
    n_envs=4,
    n_steps=128,
    batch_size=128,
    ent_coef=0.01,
    eval_interval=100_000,
    n_evals=1,
    n_eval_envs=1,
    learning_rate=2.5e-4,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.1,
    verbose_ppo=1,
    log_interval=50,
)

# Agent hyperparameters — aligned with EvoGym paper (Table 2) and baseline GA script.
ENV_NAME = "UpStepper-v0"
STRUCTURE_SHAPE = (5, 5)
N_DESIGNS_PER_GEN = 25           # paper population size
N_ELITE_PARENTS = 5              # number of top elites carried forward and used as mutation parents
ELITE_SIZE = 5
N_LLM_SLOTS = 15                 # Path A: LLM skill-guided mutation slots per generation
N_GA_SLOTS = 10                  # Path G: GA random mutation slots per generation (overflow from LLM adds here)
MUTATION_RANGE = "1-3"           # LLM mutation range: fixed, no phase split (GA handles global exploration)
SKILL_WEIGHT_DELTA_CAP = 2.0     # diff >= cap counts as full positive contribution for skill sampling
NUM_CORES = 25                   # match baseline --num-cores 25
# 20 gens × 25 designs = 500 evaluations — matches baseline --max-evaluations 500.
TOTAL_GENERATIONS = 20
MAX_EVALUATIONS = 500  # hard cap on total PPO evaluations (0 = unlimited)

# LLM
LLM_MODEL = "gpt-5.5"
LLM_MODEL_STRONG = "gpt-5.5"  # Used for consolidation clustering + merge + new insight
LLM_TEMPERATURE = 0.7
# API key fallback — prefer setting OPENAI_API_KEY env var to avoid accidental git commits
OPENAI_API_KEY = ""  # fill in if not using env var

# v2: Pool re-attribution threshold (re-run attribution on accumulated pool when it grows large)
RE_ATTRIBUTION_POOL_THRESHOLD = 30

MAX_LEAF_DESCRIPTION_WORDS = 100
