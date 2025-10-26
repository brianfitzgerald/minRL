from tqdm import tqdm
from minrl.constants import Conversation
from minrl.packing import pack_bfd
from minrl.tasks.dataset import MinRLDataset
from minrl.trainer import Trainer
import torch
import torch.nn.functional as F
from loguru import logger
from minrl.utils import log_memory_usage, clear_memory
from minrl.algorithms import get_token_ids_and_assistant_mask
from torch.utils.data import DataLoader
from typing import Any, cast
import time
from contextlib import nullcontext


class SFTTrainer(Trainer):
    def init_model(self):
        """Initialize model without vLLM (not needed for SFT)."""
        self._setup_hf_model()

    def init_training(self) -> None:
        """Initialize training components."""
        from minrl.tasks import TASK_DATASETS

        assert self.tokenizer is not None, "Tokenizer not initialized"
        dataset_cls = TASK_DATASETS[self.config.task]
        self.reward_function = dataset_cls.reward_function
        self.train_dataset = dataset_cls(
            split="train", host=self.host_type, tokenizer=self.tokenizer
        )
        self.eval_dataset = dataset_cls(
            split="eval", host=self.host_type, tokenizer=self.tokenizer
        )

        # Get pad token ID
        self.pad_token_id = int(cast(Any, self.tokenizer.pad_token_id))

        # Create dataloader with sequence packing collate function
        def collate_fn(samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return pack_bfd(
                samples,
                self.config.max_seq_length,
            )

        generator = torch.Generator(device=self.device)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            generator=generator,
            batch_size=self.config.prompts_per_full_batch,
            collate_fn=collate_fn,  # type: ignore
            pin_memory=False,
            num_workers=0,
        )
        self.start_time = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Batch size: {self.config.prompts_per_full_batch}, micro batch size: {self.config.micro_batch_size}"
        )

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.model is not None

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward pass with mixed precision
        fwd_ctx = (
            torch.autocast(device_type="cuda", enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with fwd_ctx:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # Compute cross-entropy loss
            # labels already have -100 for non-target tokens
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

        # Compute metrics
        with torch.no_grad():
            # Count target tokens (not -100)
            n_target_tokens = (labels != -100).sum().item()

            # Compute perplexity
            perplexity = torch.exp(loss).item() if loss.item() < 20 else float("inf")

            # Compute accuracy on target tokens
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels) & (labels != -100)
            accuracy = correct.sum().item() / max(n_target_tokens, 1)

        metrics = {
            "loss": loss.item(),
            "perplexity": perplexity,
            "accuracy": accuracy,
            "n_target_tokens": n_target_tokens,
        }

        return loss, metrics

    def train(self) -> None:
        """Run the main SFT training loop.

        For each step:
            1. Loads batch of training data with sequence packing
            2. Computes loss on assistant tokens only
            3. Performs gradient accumulation over micro-batches
            4. Evaluates model periodically
            5. Saves checkpoints periodically
            6. Logs metrics to TensorBoard or Weights & Biases
        """
        assert self.model is not None
        assert self.tokenizer is not None

        self.model.train()
        global_step = 0

        for epoch in range(100):  # Large number, will be stopped manually
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch + 1}")

            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}"):
                step_start_time = time.time()
                global_step += 1

                # Compute loss
                loss, metrics = self.compute_loss(batch)

                # Backward pass
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Log metrics
                self.metrics_wrapper.add_scalar(
                    "train/loss", metrics["loss"], global_step
                )
                self.metrics_wrapper.add_scalar(
                    "train/perplexity", metrics["perplexity"], global_step
                )
                self.metrics_wrapper.add_scalar(
                    "train/accuracy", metrics["accuracy"], global_step
                )
                self.metrics_wrapper.add_scalar(
                    "train/grad_norm", float(grad_norm), global_step
                )
                self.metrics_wrapper.add_scalar(
                    "train/n_target_tokens", metrics["n_target_tokens"], global_step
                )
                self.metrics_wrapper.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], global_step
                )

                # Log timing
                step_duration = time.time() - step_start_time
                self.metrics_wrapper.add_scalar(
                    "timing/train_step_sec", step_duration, global_step
                )

                # Log memory
                log_memory_usage(
                    "train_step", metrics_wrapper=self.metrics_wrapper, step=global_step
                )

                # Clear memory
                clear_memory()

                # Evaluate periodically
                if global_step % self.config.eval_interval == 0:
                    self.evaluate(global_step)
                    self.model.train()

                # Save checkpoint periodically
                if global_step % self.config.ckpt_save_interval == 0:
                    logger.info(f"Saving checkpoint at step {global_step}")
                    output_file = (
                        self.checkpoint_dir / f"{self.run_name}_step_{global_step:06d}"
                    )
                    self.model.save_pretrained(output_file)  # type: ignore
                    logger.info(f"Saved checkpoint to {output_file}")

                # Log progress
                if global_step % 10 == 0:
                    logger.info(
                        f"Step {global_step} | Loss: {metrics['loss']:.4f} | "
                        f"PPL: {metrics['perplexity']:.2f} | Acc: {metrics['accuracy']:.3f} | "
                        f"Grad norm: {grad_norm:.3f}"
                    )

            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s")

        self.metrics_wrapper.close()

    def evaluate(self, step: int) -> None:
        """Evaluate the current model by generating responses and computing rewards.

        Args:
            step: Current training step for logging
        """
        assert self.model is not None
        assert self.tokenizer is not None

        logger.info(f"Starting evaluation at step {step}")
        self.model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            collate_fn=lambda x: x,  # No packing for eval
        )

        all_rewards = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Generate responses
                conversations = [
                    self.eval_dataset.initial_conversation(sample, i)
                    for i, sample in enumerate(batch)
                ]

                # For evaluation, we can either:
                # 1. Compute loss on ground truth (teacher forcing)
                # 2. Generate responses and compute rewards

                # Option 1: Compute loss on ground truth
                packed_batch = pack_bfd(
                    batch,
                    self.config.max_seq_length,
                )

                _, metrics = self.compute_loss(packed_batch)
                total_loss += metrics["loss"]
                num_batches += 1

                # Option 2: Generate and compute rewards
                # Remove the last assistant message to generate from scratch
                for conversation, sample in zip(conversations, batch):
                    # Generate response
                    prompt = [msg for msg in conversation[:-1]]  # Remove ground truth

                    templated_prompt = self.tokenizer.apply_chat_template(
                        prompt,  # type: ignore
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = self.tokenizer(
                        templated_prompt,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                    # Generate with greedy decoding
                    outputs = cast(Any, self.model).generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=1.0,
                        do_sample=False,  # Greedy
                        pad_token_id=self.pad_token_id,
                    )

                    # Decode generated text
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )

                    # Create conversation with generated response
                    eval_conversation = prompt + [
                        {"role": "assistant", "content": generated_text}
                    ]

                    # Compute reward
                    reward = self.reward_function(
                        cast(Conversation, eval_conversation), sample
                    )
                    all_rewards.append(reward)

        # Compute metrics
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        mean_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log metrics
        self.metrics_wrapper.add_scalar("eval/mean_reward", mean_reward, step)
        self.metrics_wrapper.add_scalar("eval/loss", mean_loss, step)

        logger.info(
            f"Evaluation complete | Mean reward: {mean_reward:.4f} | Mean loss: {mean_loss:.4f}"
        )

        clear_memory()
