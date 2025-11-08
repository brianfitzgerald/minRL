import fire

from minrl.trainer import Trainer


def main(wandb: bool = False) -> None:
    trainer = Trainer("local", "wandb" if wandb else "tensorboard")
    trainer.init_model()
    trainer.init_training()
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
