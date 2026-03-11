# Codebase File Guide

This README documents every `.py`, `.ipynb`, and `.csv` file currently present in this workspace and its subdirectories.

## Root Directory (`./`)

### Python Files

| File | Description |
| --- | --- |
| `calc_performance3.py` | Serial policy evaluation helper. Runs multiple episodes in a single process and returns average reward for a policy (supports deterministic or stochastic action selection). |
| `calc_performance3_parallel.py` | Parallel version of policy evaluation using `torch.multiprocessing` workers. Splits evaluation episodes across workers and aggregates total reward. |
| `call_paramset.py` | Utilities for experiment setup: `call_paramset()` parses a hyperparameter CSV row and expands semicolon-separated tuning values into combinations; `call_env()` instantiates environments from config dictionaries. |
| `dispersal_weight_generator.py` | Generates patch-to-patch dispersal matrices (uniform random spatial layout with exponential or Gaussian kernels) and optionally normalizes connectivity. Script mode saves many sampled matrices/coordinates to `./dispersal_weights/`. |
| `FixedMeanStd.py` | Observation normalization helper. In current `metapop1` usage, it mainly normalizes time by horizon `T` (with support for 1D or 2D state layouts). |
| `metapop1.py` | Core metapopulation environment class (`metapop1`). Loads environment settings/parameters from CSV, defines state/action spaces, simulates transitions, rewards, termination, and supports partial observability and optional 2D state representation with connectivity features. |
| `metapop_value_iteration.py` | Exact finite-horizon dynamic programming (value iteration) utilities for fully observable `metapop1` setups. Builds optimal controller tables and action decoding helpers (`_act`, `_act_from_components`). |
| `ppo_actor.py` | Actor network for PPO in `metapop1` MDP settings. Supports Bernoulli action heads and without-replacement sampling with STOP tokens for constrained patch action selection. |
| `ppo_critic.py` | Critic network (value function approximator) used by PPO. Configurable MLP with optimizer and LR scheduler setup. |
| `ppo_actorcritic_encoder.py` | Shared actor-critic architecture with an encoder for PPO2. Encodes patch-wise inputs, pools global context, produces action logits and state value estimates in one model. |
| `ppoagent.py` | PPO training agent implementation with rollout memory, GAE computation, PPO clipped objective, optional entropy loss, and optional KL early stopping. Uses separate actor/critic networks. |
| `ppoagent2.py` | PPO2 agent implementation for shared actor-critic models. Similar PPO training flow but uses the combined encoder/actor/critic network pathway. |
| `PPO.py` | End-to-end training runner class for PPO experiments. Parses hyperparameters, creates actor/critic + `PPOAgent`, handles training loop, periodic evaluation, checkpointing, and result export to `./PPO_results/`. |
| `PPO2.py` | End-to-end training runner class for PPO2 (shared actor-critic with encoder). Mirrors PPO workflow with PPO2-specific model/agent wiring and output management. |
| `setup_logger.py` | Logging utility that creates timestamped run folders and redirects `print()` output to both console and `output.log`. |

### Notebooks

| File | Description |
| --- | --- |
| `dispersal_weights_exp.ipynb` | Exploratory notebook for dispersal-weight modeling. Samples generated weight matrices across patch counts and plots distributions of top incoming weights, total connectivity, and edge weights. |
| `env_test.ipynb` | Sanity-check notebook for environment behavior and value-iteration controller generation. Includes manual stepping in `metapop1` and saving a value-iteration controller pickle. |
| `performance_testing.ipynb` | Evaluation notebook comparing policy behavior (value iteration and heuristic baselines) over many episodes and reporting average rewards. |
| `ppotrain.ipynb` | Main interactive training notebook with two sections: PPO training and PPO2 training loops over seeds/hyperparameter sets. |
| `ppotrain copy.ipynb` | Copy of `ppotrain.ipynb` with the same two training sections (PPO and PPO2), likely used as a backup or alternate experiment draft. |

### CSV Files

| File | Description |
| --- | --- |
| `envsetting.csv` | Environment setting catalog keyed by `ID`. Contains scenario-level switches such as observability, patch count, action portfolio, dispersal regime/type, parameter set ID, action limits (`kR`, `kS`), and state-shape options. |
| `metapop_paramset.csv` | Metapopulation dynamics parameter table keyed by `paramsetID`, including habitat degradation, dispersal regime transition probabilities, colonization/extinction coefficients, survey detection errors, horizon, budget, and terminal penalty scale. |

## Subdirectory: `hyperparamsets/`

### CSV Files

| File | Description |
| --- | --- |
| `hyperparamsets/PPOhyperparamsets.csv` | Hyperparameter matrix for PPO experiments. Rows define network sizes, learning rates/schedulers, PPO loss settings, rollout/minibatch configuration, evaluation schedule, and related training flags. |
| `hyperparamsets/PPO2hyperparamsets.csv` | Hyperparameter matrix for PPO2 experiments (shared encoder actor-critic), including encoder architecture/lr in addition to PPO training settings. |

## Subdirectory: `PPO_results/`

No `.py`, `.ipynb`, or `.csv` files were found under this directory at the time of documentation. This folder currently appears to store experiment artifacts (for example, model checkpoints, logs, arrays, and config text files).

## Subdirectory: `value_iteration/`

No `.py`, `.ipynb`, or `.csv` files were found under this directory at the time of documentation. This folder is referenced by notebooks/scripts for saved value-iteration controllers (pickle files).

## Subdirectory: `dispersal_weights/`

No `.py`, `.ipynb`, or `.csv` files were found under this directory at the time of documentation. This folder is used for generated dispersal weight/coordinate pickle files.

## Notes

- This inventory reflects the current workspace state as scanned on 2026-03-06.
- If new experiment scripts or data tables are added in nested result folders, update this README so the catalog stays complete.
