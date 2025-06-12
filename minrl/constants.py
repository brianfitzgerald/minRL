from pydantic import BaseModel
from typing import Literal

QWEN_3_0_6B = "Qwen/Qwen3-0.6B"
QWEN_3_1_7_B = "Qwen/Qwen3-1.7B"
QWEN_25_05B = "Qwen/Qwen2.5-0.5B-Instruct"

OptimizerChoice = Literal["adamw", "adamw_8bit"]

AlgorithmChoice = Literal["reinforce", "grpo"]


class TrainerConfig(BaseModel):
    model_id: str = QWEN_3_0_6B
    eval_interval: int = 100
    num_answers_per_question: int = 4
    max_new_tokens: int = 1024
    micro_batch_size: int = 2
    max_grad_norm: float = 0.01
    ckpt_save_interval: int = 50
    skip_unfinished_episodes: bool = False
    optimizer: OptimizerChoice = "adamw"
    algorithm: AlgorithmChoice = "reinforce"
    lr: float = 1e-5
