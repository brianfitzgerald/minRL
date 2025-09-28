from minrl.algorithms import update_policy
from minrl.constants import Conversation, Episode, TrainerConfig
from minrl.trainer import Trainer

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

policy_results = update_policy(
    model=trainer.model,  # pyright: ignore[reportArgumentType]
    optimizer=trainer.optimizer,
    episodes=episodes,
    micro_batch_size=trainer.config.train_batch_size,
    pad_token_id=int(trainer.tokenizer.pad_token_id),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
    max_grad_norm=trainer.config.max_grad_norm,
    device=trainer.device,
    algorithm=trainer.config.algorithm,
    tokenizer=trainer.tokenizer,  # pyright: ignore[reportArgumentType]
    apply_loss=True,
)

print(policy_results)


def main(compute_logprobs: bool = False):
    pass


if __name__ == "__main__":
    fire.Fire(main)
