import fire
from minrl.trainer import Trainer


def main() -> None:
    """Main entry point for training.

    Args:
        model_id: The HuggingFace model ID to use for training
    """
    trainer = Trainer("local")
    trainer.init_model()
    trainer.init_training()
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
