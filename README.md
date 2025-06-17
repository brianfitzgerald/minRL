# MinRL

Minimal reimplementation of GRPO and REINFORCE, and other RL algorithms for LLM training.

Attempts to reuse some aspects of TRL's vLLM integration.

Also references GRPO-Zero repo somewhat - but will likely replace over time.


## Commands


To run:

```bash
python train.py
```

Logs:
```bash
tensorboard --logdir runs
```

Modal:
```bash
uv run modal run -d modal_train.py::training
```