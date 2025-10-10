from dataclasses import dataclass
from typing import Any, Callable, Literal, NotRequired, Required, TypeAlias, TypedDict


class StepMetadata(TypedDict):
    observation: str
    inventory: str
    score: int
    moves: int
    location: str
    reward: float


class ConversationMessage(TypedDict, total=False):
    role: Required[Literal["system", "user", "assistant"]]
    content: Required[str]
    reasoning: NotRequired[str | None]
    step_metadata: NotRequired[StepMetadata | None]


HIGH_QUALITY_GAMES = ["temple.z5", "zork1.z5", "zork2.z5", "zork3.z5"]

Conversation: TypeAlias = list[ConversationMessage]


Sample: TypeAlias = dict[str, Any]
RewardFunction: TypeAlias = Callable[[Conversation, Sample], float]


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    # Index of group in batch
    group_index: int
    # Index of answer in group
    answer_index: int
    # Whether the episode finished or terminated early
    reward: float
    conversation: Conversation
    sample: Sample
    # Cache tokenization results to avoid re-tokenizing
    _token_ids: list[int] | None = None
    _assistant_mask: list[bool] | None = None


QWEN_3_0_6B = "Qwen/Qwen3-0.6B"
QWEN_3_1_7_B = "Qwen/Qwen3-1.7B"
QWEN_25_05B = "Qwen/Qwen2.5-0.5B-Instruct"
TINY_LLAMA_V0 = "Maykeye/TinyLLama-v0"
SMOL_LM_2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOL_LM_2_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
GEMMA_3_270M = "google/gemma-3-270m-it"
GEMMA_TINY_TESTING = "trl-internal-testing/tiny-GemmaForCausalLM"


OptimizerChoice = Literal["adamw"]
AlgorithmChoice = Literal["reinforce", "grpo", "gpg"]
LoggerChoice = Literal["tensorboard", "wandb"]

HostType = Literal["modal", "local"]
DeviceType = Literal["cuda", "mps", "cpu"]
TaskChoice = Literal["connections", "hanoi", "zork"]
ModelType = Literal["openrouter", "openai", "huggingface", "finetuned"]

EvalSampleStatus = Literal["running", "done", "error"]


class EvalSample(TypedDict):
    model: str
    conversation: Conversation
    status: EvalSampleStatus
    game: NotRequired[str]


class EvalModel(TypedDict):
    type: ModelType
    model_id: str
    base_model_id: NotRequired[str]


ModelName = Literal[
    "gemini_2_flash",
    "gpt-4.1-mini",
    "Qwen3.0-6B",
    "qwen_grpo",
    "qwen_reinforce",
    "magistral_medium",
    "gpt-4o",
    "o4_mini",
    "gemini_2.5_flash",
]

INFERENCE_MODELS: dict[ModelName, EvalModel] = {
    "gemini_2_flash": {
        "type": "openrouter",
        "model_id": "google/gemini-2.0-flash-001",
    },
    "gemini_2.5_flash": {
        "type": "openrouter",
        "model_id": "google/gemini-2.5-flash",
    },
    "gpt-4.1-mini": {"type": "openai", "model_id": "gpt-4.1-mini"},
    "gpt-4o": {"type": "openai", "model_id": "gpt-4o-2024-08-06"},
    "o4_mini": {"type": "openai", "model_id": "o4-mini"},
    "Qwen3.0-6B": {"type": "huggingface", "model_id": QWEN_3_0_6B},
    "qwen_grpo": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-grpo-connections-0620_221447_step_003500",
        "base_model_id": QWEN_3_0_6B,
    },
    "qwen_reinforce": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-reinforce-20250612_213402_step_000900",
        "base_model_id": QWEN_3_0_6B,
    },
    "magistral_medium": {
        "type": "openrouter",
        "model_id": "mistralai/magistral-medium-2506:thinking",
    },
}


@dataclass
class TrainerConfig:
    model_id: str = GEMMA_3_270M
    eval_interval: int = 10
    # Total batch size is groups_per_batch * group_size
    groups_per_batch: int = 4
    group_size: int = 4
    max_new_tokens: int = 512
    eval_batch_size: int = 8
    max_grad_norm: float = 1.0
    ckpt_save_interval: int = 500
    lr: float = 5e-6
    optimizer: OptimizerChoice = "adamw"
    use_low_precision_optimizer_if_available: bool = True
    algorithm: AlgorithmChoice = "grpo"
    task: TaskChoice = "connections"
    wandb_project: str = "minrl"
    wandb_entity: str | None = None
    temperature: float = 1.2
    temperature_scaling: bool = False
    temperature_min: float = 0.2
    temperature_max: float = 1.5
    entropy_coef: float = 0.01  # Entropy regularization coefficient

    # Memory optimization settings
    use_gradient_checkpointing: bool = True
    micro_batch_size: int = 2  # Size of micro-batches

    @property
    def model_display_name(self) -> str:
        return self.model_id.split("/")[-1].replace("-", "_")
