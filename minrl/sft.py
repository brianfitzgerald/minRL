from minrl.trainer import Trainer
import torch
from loguru import logger
from minrl.utils import log_memory_usage
from torch.utils.data import DataLoader
import time


class SFTTrainer(Trainer):
    def init_model(self):
        self._setup_hf_model()

    def init_training(self) -> None:
        generator = torch.Generator(device=self.device)
        # Reduce batch size for memory efficiency
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            generator=generator,
            batch_size=self.config.groups_per_batch,
            collate_fn=lambda x: x,
            pin_memory=False,  # Disable pin_memory to save memory
            num_workers=0,  # Use single process to avoid memory overhead
        )
        self.start_time = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run the main training loop.

        For each step:
            1. Performs rollout to generate episodes
            2. Updates policy using GRPO
            3. Evaluates model periodically
            4. Saves checkpoints periodically
            5. Logs metrics to TensorBoard or Weights & Biases
        """

        prev_reward_std: float | None = None

        for step, batch in enumerate(self.train_dataloader, start=1):
            step_start_time = time.time()
            logger.info(f"Starting rollout for step {step}")

            assert self.model is not None
            assert self.tokenizer is not None

            conversations = [
                self.train_dataset.initial_conversation(sample, i)
                for i, sample in enumerate(batch)
            ]
            # Log memory usage after clearing
            log_memory_usage(
                "end_of_step", metrics_wrapper=self.metrics_wrapper, step=step
            )
            if step % self.config.eval_interval == 0:
                self.evaluate(step)

            # Update prev_reward_std for next iteration
            prev_reward_std = current_reward_std
            # save checkpoint
            if step % self.config.ckpt_save_interval == 0:
                logger.info(f"Saving checkpoint for step {step}")
                output_file = self.checkpoint_dir / f"{self.run_name}_step_{step:06d}"
                self.model.save_pretrained(output_file)  # type: ignore
                logger.info(f"Saved checkpoint to {output_file}")

            # Log total step duration
            total_step_duration = time.time() - step_start_time
            self.metrics_wrapper.add_scalar(
                "timing/total_step_duration_sec", total_step_duration, step
            )
            logger.info(f"Step {step} completed in {total_step_duration:.2f}s")

        self.metrics_wrapper.close()

    def evaluate(self, step: int) -> None:
        """Evaluate the current model.

        Returns:
            float: The evaluation success rate (currently returns 0.0 as placeholder)
        """
        eval_loader = DataLoader(
            self.eval_dataset,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            collate_fn=lambda x: x,
        )
        assert self.model is not None

        assert self.tokenizer is not None
        mean_reward = 0
        episodes: list[Episode] = []
        for batch in tqdm(eval_loader):
            conversations = [
                self.eval_dataset.initial_conversation(sample, i)
                for i, sample in enumerate(batch)
            ]
            batch_episodes, rollout_duration = rollout(
                self.config,
                self.tokenizer,
                1,
                self.eval_dataset.max_steps,
                conversations,
                samples=batch,
                reward_function=self.reward_function,
                vllm_model=self.vllm_model,
            )

            self.metrics_wrapper.add_scalar(
                "timing/eval_rollout_duration_sec", rollout_duration, step
            )

            episodes.extend(batch_episodes)

        reward = [episode.reward for episode in episodes]
        mean_reward = sum(reward) / len(reward)
        self.metrics_wrapper.add_scalar("eval/mean_reward", mean_reward, step)
