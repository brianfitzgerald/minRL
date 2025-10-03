from minrl.algorithms import (
    get_token_ids_and_assistant_mask,
    process_batch,
    update_policy,
)
from minrl.constants import AlgorithmChoice, Conversation, Episode, TrainerConfig
from minrl.trainer import Trainer
import torch

import fire


config = TrainerConfig(train_batch_size=4)

trainer = Trainer(host_type="local")
trainer.init_model()
conversations: list[Conversation] = [
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hey!"},
    ],
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "What is up?"},
    ],
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "What is your name?"},
    ],
]

episodes = [
    Episode(
        group_index=0,
        answer_index=0,
        sample={"id": 0, "prompt": "Hello"},
        reward=1.0,
        conversation=conversations[0],
    ),
    Episode(
        group_index=0,
        answer_index=1,
        sample={"id": 0, "prompt": "Hello"},
        reward=0.5,
        conversation=conversations[1],
    ),
    Episode(
        group_index=0,
        answer_index=2,
        sample={"id": 0, "prompt": "Hello"},
        reward=0.2,
        conversation=conversations[2],
    ),
    Episode(
        group_index=0,
        answer_index=3,
        sample={"id": 0, "prompt": "Hello"},
        reward=0.3,
        conversation=conversations[3],
    ),
]


def compute_algorithm_loss(
    logprobs: torch.Tensor,
    target_masks: torch.Tensor,
    batch_rewards: torch.Tensor,
    algorithm: AlgorithmChoice,
    n_target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # multiply the log probs by the advantages
    if algorithm == "grpo":
        advantage_t = logprobs * batch_rewards[:, None]
    elif algorithm == "gpg":
        # subtract baseline, which is the mean of the rewards
        advantages = batch_rewards - batch_rewards.mean()
        advantage_t = logprobs * advantages[:, None]
    elif algorithm == "reinforce":
        advantage_t = logprobs * batch_rewards[:, None]

    # scale by the mask, and normalize by token count
    # this sets the advantage to 0 for padding tokens
    advantage_t = (advantage_t * target_masks).sum() / n_target_tokens

    loss = -advantage_t

    return loss, advantage_t


def main():
    n_target_tokens = 0
    for episode in episodes:
        token_ids, _ = get_token_ids_and_assistant_mask(
            episode.conversation, trainer.tokenizer
        )
        n_target_tokens += len(token_ids)

    # Process the batch
    logprobs, target_msks, batch_rewards_t, batch_entropy = process_batch(
        model=trainer.model,
        episodes=episodes,
        tokenizer=trainer.tokenizer,
        pad_token_id=int(trainer.tokenizer.pad_token_id),  # pyright: ignore[reportArgumentType]
        device=trainer.device,
        n_target_tokens=n_target_tokens,
    )
    batch_loss, advantage_t = compute_algorithm_loss(
        logprobs,
        target_msks,
        batch_rewards_t,
        algorithm="grpo",
        n_target_tokens=n_target_tokens,
    )

    print(batch_loss, advantage_t)


if __name__ == "__main__":
    fire.Fire(main)
