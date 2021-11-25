import argparse
import os
from re import I

from transformers import pipeline
from tqdm import tqdm

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


def main():
    config = get_config()
    dataset = get_dataset(dataset=config.dataset, download=False)

    # TODO: Transform the data so that it can be passed into the BERT model.
    train_data = dataset.get_subset(TRAIN_SPLIT, frac=config.frac)
    train_loader = get_train_loader(config.train_loader, train_data, batch_size=16)
    for x, y_true, metadata in train_loader:
        # Fine-tune the HuggingFace model on all the training data and save it.
        print(f"{x}, {y_true}, {metadata}")

    # Sample data from the eval set using the following critiera:
    val_data = dataset.get_subset(VAL_SPLIT, frac=config.frac)
    val_loader = get_eval_loader(config.loader, val_data, batch_size=100)

    # Random.
    # Uncertainty.
    # Self-Learning.
    # Traffic-Aware Uncertainty.

    # At each round, evalute on all of the test data.

    # Get model predictions
    classifier = pipeline("sentiment-analysis")
    prediction = classifier("Wow, using HuggingFace is easy")
    print(prediction)


def get_config():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "-d", "--dataset", choices=["amazon", "civilcomments"], required=True
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).",
    )
    # Dataset
    parser.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.",
    )
    # Loaders
    parser.add_argument(
        "--train_loader", choices=["standard", "group"], default="standard"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loader", choices=["standard"], default="standard")

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    main()
