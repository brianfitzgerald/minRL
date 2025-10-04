

## Development Commands

Always activate the virtualenv first with `source .venv/bin/activate`.

### Training
```bash
# Run training locally
python train.py

# Run training on Modal
uv run modal run -d modal_train.py::training

# Run tests
uv run pytest tests/

# Lint code
ruff check .
```

### Testing
```bash
python tests/
```

## Architecture

### Core Components

1. **Trainer** (`minrl/trainer.py`): Main training orchestrator
   - Manages model initialization with vLLM and transformers
   - Handles training loop with rollout → policy update → evaluation cycle
   - Supports both local and Modal deployment modes
   - Integrates with TensorBoard and Weights & Biases for logging

2. **Algorithms** (`minrl/algorithms.py`): RL algorithm implementations
   - `rollout()`: Generates responses using vLLM for inference
   - `update_policy()`: Updates policy using GRPO, REINFORCE, or GPG algorithms
   - `normalize_rewards_per_group()`: Normalizes rewards within response groups
   - Temperature scaling based on reward standard deviation

3. **Tasks** (`minrl/tasks/`): Task definitions and datasets
   - Each task has a reward function and dataset class
   - Tasks are registered in `TASK_DEFINITIONS` dictionary

4. **Configuration** (`minrl/constants.py`): Centralized configuration
   - `TrainerConfig`: Main configuration with model settings, hyperparameters
   - Model definitions for various sizes (SmolLM, Qwen variants)
   - Algorithm choices: "reinforce", "grpo", "gpg"