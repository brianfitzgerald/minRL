import fire
from minrl.trainer import Trainer


def main() -> None:
    trainer = Trainer("local")
    trainer.init_model()
    trainer.init_training()
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
