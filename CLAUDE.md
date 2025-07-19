# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MinRL is a minimal reimplementation of GRPO (Group Relative Policy Optimization) and other reinforcement learning algorithms for LLM training. The project uses PyTorch and vLLM for model training and inference, with support for various tasks like Connections and Hanoi Tower puzzles.

## Development Commands

### Training
```bash
# Run training locally
python train.py

# Run training on Modal
uv run modal run -d modal_train.py::training

# Run tests
pytest tests/

# Lint code
ruff check .
```

### Monitoring
```bash
# View TensorBoard logs
tensorboard --logdir runs

# Evaluation visualization
python visualize_evals.py
```

### Testing
```bash
# Run specific test files
pytest tests/test_connections_reward.py
pytest tests/test_hanoi.py

# Run a quick test
python test_run.py
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
   - `ConnectionsDataset`: NYT Connections puzzle task
   - `HanoiDataset`: Tower of Hanoi puzzle task
   - Tasks are registered in `TASK_DEFINITIONS` dictionary

4. **Configuration** (`minrl/constants.py`): Centralized configuration
   - `TrainerConfig`: Main configuration with model settings, hyperparameters
   - Model definitions for various sizes (SmolLM, Qwen variants)
   - Algorithm choices: "reinforce", "grpo", "gpg"

### Key Design Patterns

- **Dual Model Architecture**: Uses both transformers (for training) and vLLM (for inference)
- **Episode-based Training**: Groups responses by prompt prefix for reward normalization
- **Modular Task System**: Easy to add new tasks by implementing reward function and dataset
- **Flexible Deployment**: Supports both local training and Modal cloud deployment

### Dependencies

- **Core ML**: `torch`, `transformers`, `vllm`, `accelerate`
- **Data**: `datasets`, `pandas`, `numpy`
- **Logging**: `tensorboard`, `wandb`, `loguru`
- **Testing**: `pytest`
- **Deployment**: `modal` (for cloud training)

## Configuration

The main configuration is in `TrainerConfig` class in `constants.py`:
- `model_id`: HuggingFace model identifier
- `algorithm`: RL algorithm choice ("grpo", "reinforce", "gpg")
- `task`: Task to train on ("connections", "hanoi")
- `train_batch_size`: Training batch size
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (supports adaptive scaling)

## File Structure

- `minrl/`: Main package
  - `trainer.py`: Training orchestration
  - `algorithms.py`: RL algorithms
  - `constants.py`: Configuration classes
  - `metrics.py`: Metrics logging wrapper
  - `tasks/`: Task implementations
- `train.py`: Local training entry point
- `modal_train.py`: Modal deployment script
- `evaluate.py`: Evaluation utilities
- `tests/`: Test suite
- `runs/`: TensorBoard logs
- `checkpoints/`: Model checkpoints