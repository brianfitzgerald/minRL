import modal
from modal import App, Image

from minrl.modal_utils import (
    DATASET_VOLUME,
    DATASET_VOLUME_PATH,
    MODEL_WEIGHTS_VOLUME,
    MODELS_VOLUME_PATH,
)
from minrl.trainer import Trainer

APP_NAME = "minRL"
app = App(APP_NAME)


CUDA_VERSION = "12.4.0"  # should be no greater than host CUDA version
FLAVOR = "devel"  #  includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
TAG = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"

MODAL_IMAGE = (
    Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.12")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/uv.lock", copy=True)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .apt_install("git", "build-essential", "ninja-build")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": MODELS_VOLUME_PATH.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    # Prefer GCC toolchain for PyTorch C++/CUDA extensions
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands(
        [
            "uv sync --no-group training --no-group games",
            "uv sync --group training --no-group games --no-build-isolation",
        ]
    )
    .add_local_python_source(
        "minrl",
    )
    .add_local_file(".env", "/.env")
    .add_local_dir("data", "/data")
)


def format_timeout(seconds: int = 0, minutes: int = 0, hours: int = 0):
    return seconds + (minutes * 60) + (hours * 60 * 60)


@app.function(
    image=MODAL_IMAGE,
    gpu="A100-40GB:1",
    secrets=[
        modal.Secret.from_name("smolmodels"),
    ],
    volumes={
        MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME,
        DATASET_VOLUME_PATH.as_posix(): DATASET_VOLUME,
    },
    timeout=format_timeout(hours=6),
)
def training():
    trainer = Trainer("modal", "wandb")
    trainer.init_model()
    trainer.init_training()
    trainer.train()
