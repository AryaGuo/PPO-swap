import os
import time

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model import PPOLightning
from utils import get_config


def train_ppo(config):
    log_path = "./logs/"
    try:
        log_name = config["log_name"]
    except KeyError:
        log_name = time.strftime("%Y%m%d-%H%M%S")

    if os.path.exists(f"{log_path}/{log_name}/checkpoints"):
        print(f"{log_path}/{log_name} already exists")
        return

    os.makedirs(f"{log_path}/{log_name}", exist_ok=True)
    yaml.safe_dump(config, open(f"{log_path}/{log_name}/config.yaml", "w"))
    print("config saved at", f"{log_path}/{log_name}/config.yaml")

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_path, name="", version=log_name, default_hp_metric=False
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="hp/avg_ep_reward", mode="max", save_last=True
    )
    trainer = Trainer(
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=tb_logger,
        **config["ppo_trainer"],
    )
    model = PPOLightning(**config["ppo"])
    trainer.fit(model)


if __name__ == "__main__":
    config = get_config(["-c", "config/train.yaml"])
    train_ppo(config)