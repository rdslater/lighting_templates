import argparse
from pathlib import Path

import torch
import yaml

from model import Classifier


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


def find_checkpoint(experiment_name):
    checkpoint_path = (
        Path("outputs")
        / experiment_name
        / "checkpoints"
        / "best.ckpt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path


def main():
    args = parse_args()
    cfg = load_config(args.config)

    checkpoint_path = find_checkpoint(
        cfg["experiment_name"]
    )

    print(f"Loading: {checkpoint_path}")

    model = Classifier.load_from_checkpoint(
        checkpoint_path
    )

    model.eval()

    # Example input
    x = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        logits = model(x)
        prediction = logits.argmax(dim=1)

    print(f"Prediction: {prediction.item()}")


if __name__ == "__main__":
    main()
