from minrl.algorithms import update_policy
from minrl.constants import Conversation, Episode, TrainerConfig
from minrl.trainer import Trainer

config = TrainerConfig()

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
]

update_policy(
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
