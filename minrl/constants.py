from pydantic import BaseModel
from typing import Literal


QWEN_3_0_6B = "Qwen/Qwen3-0.6B"
QWEN_3_1_7_B = "Qwen/Qwen3-1.7B"
QWEN_25_05B = "Qwen/Qwen2.5-0.5B-Instruct"
SMOL_LM_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"

OptimizerChoice = Literal["adamw", "adamw_8bit"]
AlgorithmChoice = Literal["reinforce", "grpo", "gpg"]
LoggerChoice = Literal["tensorboard", "wandb"]

HostType = Literal["modal", "local"]

TaskChoice = Literal["connections", "hanoi"]


class TrainerConfig(BaseModel):
    model_id: str = QWEN_3_0_6B
    eval_interval: int = 50
    num_answers_per_question: int = 4
    max_new_tokens: int = 1024
    train_batch_size: int = 32
    eval_batch_size: int = 4
    max_grad_norm: float = 0.1
    ckpt_save_interval: int = 500
    lr: float = 5e-6
    skip_unfinished_episodes: bool = False
    optimizer: OptimizerChoice = "adamw"
    algorithm: AlgorithmChoice = "grpo"
    task: TaskChoice = "connections"
    wandb_project: str = "minrl"
    wandb_entity: str | None = None
    logger_choice: LoggerChoice = "wandb"
    temperature: float = 1.2

    @property
    def model_display_name(self) -> str:
        return self.model_id.split("/")[-1].replace("-", "_")
