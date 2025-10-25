from tensorboardX import SummaryWriter
from minrl.constants import LoggerChoice, TaskChoice, TrainerConfig

import wandb


class MetricsWrapper:
    def __init__(
        self,
        logger_choice: LoggerChoice,
        task: TaskChoice,
        trainer_config: TrainerConfig,
        run_name: str,
    ):
        self.logger_choice = logger_choice
        self.run_name = run_name
        self.writer, self.wandb_run = None, None
        if logger_choice == "wandb":
            self.wandb_run = wandb.init(
                project="minrl",
                name=run_name,
                config=trainer_config.model_dump(),
                tags=[task, trainer_config.model_id],
            )
        else:
            self.writer = SummaryWriter(f"runs/{run_name}")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.logger_choice == "wandb":
            assert self.wandb_run is not None
            self.wandb_run.log({tag: value}, step=step)
        else:
            assert self.writer is not None
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str, step: int) -> None:
        return  # TODO: fix this
        if self.logger_choice == "wandb":
            assert self.wandb_run is not None
            self.wandb_run.log({tag: wandb.Html(text)}, step=step)
        else:
            assert self.writer is not None
            self.writer.add_text(tag, f"<pre>{text}</pre>", step)

    def close(self) -> None:
        if self.logger_choice == "wandb":
            assert self.wandb_run is not None
            self.wandb_run.finish()
        else:
            assert self.writer is not None
            self.writer.close()
