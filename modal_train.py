from modal import Image, App, Secret
import modal
import os
from pathlib import Path

from minrl.constants import TrainerConfig
from train import Trainer

APP_NAME = "minRL"
MODELS_FOLDER = "model-weights"
MODELS_VOLUME_PATH = Path(f"/{MODELS_FOLDER}")
DATASET_VOLUME_PATH = os.path.join(MODELS_VOLUME_PATH.as_posix(), "dataset_files")

app = App(
    APP_NAME,
    secrets=[
        Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
    ],
)


CUDA_VERSION = "12.4.0"  # should be no greater than host CUDA version
FLAVOR = "devel"  #  includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
TAG = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"

SMOLMODELS_IMAGE = (
    Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.11")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/uv.lock", copy=True)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .apt_install("git")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": MODELS_VOLUME_PATH.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_commands(
        [
            "uv sync",
        ]
    )
    .add_local_python_source(
        "minrl",
    )
    .add_local_file(".env", "/.env", copy=False)
)


def format_timeout(seconds: int = 0, minutes: int = 0, hours: int = 0):
    return seconds + (minutes * 60) + (hours * 60 * 60)


MODEL_WEIGHTS_VOLUME = modal.Volume.from_name(MODELS_FOLDER, create_if_missing=True)


@app.function(
    image=SMOLMODELS_IMAGE,
    gpu="A100-80GB:2",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=6),
)
def run():
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.init_model()
    trainer.init_training()
    trainer.train()
