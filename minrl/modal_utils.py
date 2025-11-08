import os
from pathlib import Path

import modal
from loguru import logger
from modal.volume import FileEntryType
from tqdm import tqdm

MODAL_MODELS_VOLUME_NAME = "minrl-models"
MODAL_DATASET_VOLUME_NAME = "minrl-datasets"

MODELS_VOLUME_PATH = Path(f"/{MODAL_MODELS_VOLUME_NAME}")
DATASET_VOLUME_PATH = Path(f"/{MODAL_DATASET_VOLUME_NAME}")


def download_checkpoint_from_modal(checkpoint_name: str):
    vol = modal.Volume.from_name(MODAL_MODELS_VOLUME_NAME)
    for file in vol.iterdir(f"checkpoints/{checkpoint_name}"):
        if file.type == FileEntryType.FILE:
            # remove checkpoints/ from the path as it's already in the path
            local_path = os.path.join(".", file.path)
            if os.path.exists(local_path):
                logger.info(f"Skipping {file.path} as it already exists locally.")
                continue
            logger.info(f"Downloading {file.path} to {local_path}")
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(local_path, "wb") as file_obj:
                    for chunk in tqdm(vol.read_file(file.path), desc="Downloading"):
                        file_obj.write(chunk)
            except Exception as e:
                logger.error(f"Error downloading {file.path}: {e}")
                raise e


MODEL_WEIGHTS_VOLUME = modal.Volume.from_name(
    MODAL_MODELS_VOLUME_NAME, create_if_missing=True
)
DATASET_VOLUME = modal.Volume.from_name(
    MODAL_DATASET_VOLUME_NAME, create_if_missing=True
)
