# MinRL

Simple, clean, heavily commented implementation of various policy gradient algorithms applied to LLMs.

General principles of this codebase:
- Heavily comment each line as an education reference, a la [nanoGPT](https://github.com/karpathy/nanoGPT).
- Unit and system tested to allow for easy re-implementation.
- Easy to follow control flow. No async inference for the time being.
- Reasonably optimized - use vLLM for inference, with the sleep API, LoRA support
- Easy to install and use, and easy to hack and add new algorithms / tasks.

Influences / references:
- TRL GRPOTrainer
- GRPO-Zero
- LoRA Without Regret
- prime-rl from Prime Intellect

## Commands

To run training:

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