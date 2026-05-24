# Auto-Robotist on EvoGym

**Auto-Robotist** is a self-evolving LLM agent for soft-robot morphology design. Instead of treating each search run as a one-off optimization, the agent distills its evaluation traces into an explicit, natural-language **skill library** that can be reused across tasks and design resolutions.

This repository builds on the [EvoGym](https://evolutiongym.github.io/) benchmark (NeurIPS 2021). The original EvoGym simulator, gym wrappers, and tasks are preserved unchanged; the contribution lives in `examples/solo_leveling/`, which implements the Auto-Robotist agent on top of EvoGym's PPO + genetic-algorithm pipeline.

---

## What's inside

| Path | Purpose |
|---|---|
| `evogym/` | EvoGym C++ simulator (pybind11) + Python wrappers (`world.py`, `sim.py`, `viewer.py`) |
| `evogym/envs/` | 32 registered locomotion / traversal / manipulation environments |
| `examples/ppo/`, `examples/ga/`, `examples/bo/`, `examples/cppn_neat/` | EvoGym's stock co-design baselines |
| `examples/solo_leveling/` | **Auto-Robotist agent** — skill library, propose / evaluate / reflect loop |
| `examples/run_solo_leveling.py` | Entry point for Auto-Robotist runs |
| `examples/run_ga.py`, `run_bo.py`, `run_cppn_neat.py`, `run_ppo.py` | Baseline runners |
| `tests/`, `tutorials/`, `docs/`, `scripts/` | EvoGym test suite, tutorials, and design notes |

---

## Method overview

Auto-Robotist alternates between **using** memory and **updating** memory:

1. **Retrieve** task- and scale-relevant skills from the library.
2. **Propose** children by asking the LLM to edit elite parents toward retrieved skills; in parallel, GA mutation generates coverage proposals.
3. **Evaluate** valid candidates in EvoGym with PPO-trained controllers.
4. **Reflect**: route each evaluation as evidence to an existing skill or to an unassigned pool, then update the library with `Add` / `Diagnose` / `Merge`.

Each skill is a three-level record:

- **L1 — archetype**: a compact structural concept (e.g. `portal-frame`).
- **L2 — rules**: positive rules (raise fitness) and negative rules (failure modes), each with running mean parent-relative gain.
- **L3 — observations**: evaluated bodies, task, scale, fitness, and proposal attribution.

Because rules name *relations among functional parts* rather than absolute voxel coordinates, a `5×5` principle such as *"support a central actuator column with a rigid side frame"* can guide a `10×10` search without any voxel upsampling.

![Teaser](images/auto_robotist/autorobotist-teaser.pdf)
![Pipeline](images/auto_robotist/autorobotist-pipeline.pdf)

---

## Results at a glance

Across seven EvoGym tasks (Walker, BridgeWalker, Balancer, Carrier, Climber, Jumper, Pusher):

- **5×5 cold-start.** Auto-Robotist matches or exceeds GA on every task and reaches GA's endpoint up to **~2×** faster (`Pusher` S = 1.95, `Carrier` S = 1.67).
- **5×5 → 10×10 transfer.** Importing the learned library beats GA on **all seven** tasks. With a reference body, `Walker` jumps from 9.31 → 11.27 (Δ = +1.96, S ≈ 3.0); a skill-only variant (no source body shown) confirms gains come from *rules*, not visual imitation.

See `images/auto_robotist/` for fitness curves, skill-library inspection, and rollout comparisons.

---

## Installation

EvoGym supports Python 3.7–3.10 on Linux, macOS, and Windows.

```bash
# Clone with submodules (required for the C++ simulator)
git clone --recurse-submodules <this-repo>
cd clean_evogym

# Linux only
sudo apt-get install xorg-dev libglu1-mesa-dev

# Build the C++ simulator + install the Python package
pip install -e .
pip install -r requirements.txt
```

Requires CMake ≥ 3.1 and a C++17 toolchain (MSVC 2017+ on Windows).

Quick sanity check:

```bash
cd examples
python gym_test.py
```

---

## Running Auto-Robotist

The agent uses OpenAI-compatible models. Provide a key either via environment variable or by filling in `examples/solo_leveling/config.py`:

```bash
export OPENAI_API_KEY=sk-...
```

```bash
cd examples

# Smoke test
python run_solo_leveling.py --exp-name smoke_test --total-gen 2 --num-cores 4

# Full run
python run_solo_leveling.py --exp-name solo_walker_001 --total-gen 50 --num-cores 12
```

Outputs land in `examples/saved_data/solo_leveling/logs/<exp_name>/`:

- `summary.json` — fitness curves and per-generation aggregates
- `designs.json` — every evaluated body with attribution
- `skill_ops.json` — `Add` / `Diagnose` / `Merge` decisions made by the agent
- `skills/` — one JSON file per skill in the evolving library

### Baselines

```bash
python run_ga.py          # Genetic algorithm
python run_bo.py          # Bayesian optimization
python run_cppn_neat.py   # CPPN-NEAT
python run_ppo.py         # PPO control on a fixed morphology
```

All baselines share the same PPO controller-training protocol and per-generation morphology budget as Auto-Robotist, so curves are directly comparable.

---

## EvoGym primer

A robot is an `n × n` grid of voxel types:

| Code | Type |
|---|---|
| 0 | EMPTY |
| 1 | RIGID |
| 2 | SOFT |
| 3 | H_ACT (horizontal actuator) |
| 4 | V_ACT (vertical actuator) |
| 5 | FIXED |

A body is valid if it has a single connected non-empty component and at least one actuator. The gym interface:

```python
import gymnasium as gym
import evogym.envs
import numpy as np

body = np.array([[3, 3], [3, 3]])  # 2x2 horizontal actuator
env = gym.make('Walker-v0', body=body, render_mode='human')
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

For low-level access (custom worlds, headless rendering), use `EvoWorld` / `EvoSim` / `EvoViewer` directly — see `tutorials/` and `evogym/world.py`.

---

## Tests

```bash
cd tests
pytest -s -v -n auto                 # full suite
pytest -s -v -n auto -m lite         # fast subset
pytest -s -v screen_free/            # headless-safe only
```

---

## Acknowledgements

- The simulator, tasks, and baseline GA/BO/CPPN-NEAT runners are from [EvoGym](https://github.com/EvolutionGym/evogym) ([Bhatia et al., NeurIPS 2021](https://arxiv.org/pdf/2201.09863)).
- The Auto-Robotist agent (`examples/solo_leveling/`) and accompanying experiments are new in this repository.
