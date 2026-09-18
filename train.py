import argparse
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from model import Classifier
from data import MNISTDataModule


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnist_baseline.yaml",
    )

    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    L.seed_everything(cfg["seed"])

    data = MNISTDataModule(**cfg["data"])
    model = Classifier(**cfg["model"])

    early_stopping = EarlyStopping(
        **cfg["early_stopping"]
    )

    checkpoint_config = cfg["checkpoint"].copy()

    checkpoint_config["dirpath"] = (
        Path("outputs")
        / cfg["experiment_name"]
        / "checkpoints"
    )

    checkpoint = ModelCheckpoint(
        **checkpoint_config
    )

    trainer = L.Trainer(
        **cfg["trainer"],
        callbacks=[
            early_stopping,
            checkpoint,
        ],
    )

    trainer.fit(model, datamodule=data)

    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
