from pydantic import BaseModel
from typing import Literal

MODAL_MODELS_VOLUME_NAME = "minrl-models"
MODAL_DATASET_VOLUME_NAME = "minrl-datasets"

QWEN_3_0_6B = "Qwen/Qwen3-0.6B"
QWEN_3_1_7_B = "Qwen/Qwen3-1.7B"
QWEN_25_05B = "Qwen/Qwen2.5-0.5B-Instruct"
SMOL_LM_2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOL_LM_2_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"


OptimizerChoice = Literal["adamw", "adamw_8bit"]
AlgorithmChoice = Literal["reinforce", "grpo", "gpg"]
LoggerChoice = Literal["tensorboard", "wandb"]

HostType = Literal["modal", "local"]

TaskChoice = Literal["connections", "hanoi"]


class TrainerConfig(BaseModel):
    model_id: str = SMOL_LM_2_360M
    eval_interval: int = 10
    num_answers_per_question: int = 4
    max_new_tokens: int = 512
    train_batch_size: int = 4
    eval_batch_size: int = 16
    max_grad_norm: float = 0.1
    ckpt_save_interval: int = 500
    lr: float = 5e-6
    skip_unfinished_episodes: bool = False
    optimizer: OptimizerChoice = "adamw"
    algorithm: AlgorithmChoice = "grpo"
    task: TaskChoice = "connections"
    wandb_project: str = "minrl"
    wandb_entity: str | None = None
    temperature: float = 1.2
    temperature_scaling: bool = True
    temperature_min: float = 0.2
    temperature_max: float = 1.5

    @property
    def model_display_name(self) -> str:
        return self.model_id.split("/")[-1].replace("-", "_")
