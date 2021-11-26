import argparse
import logging
import os
from re import I
import numpy as np


from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader, DataLoader
from typing import Any, Tuple, List
from sentence_transformers import SentenceTransformer
from utils import timed

SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
LOADER = "standard"

LOSS = "hinge"
PENALTY = "l2"
MAX_ITER = 5

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    config = get_config()
    dataset = get_dataset(dataset=config.dataset, download=False)

    train_data = dataset.get_subset(TRAIN_SPLIT, frac=config.frac)
    train_loader = get_train_loader(LOADER, train_data, batch_size=config.batch_size)
    X_train, Y_train, X_train_metadata = prepare_data(train_loader)

    logger.info(f"Starting initial model training...")
    clf = SGDClassifier(loss=LOSS, penalty=PENALTY, max_iter=MAX_ITER)
    clf.fit(X_train, Y_train)
    clf.predict(X_train)

    # Sample data from the eval set using the following critiera:
    # val_data = dataset.get_subset(VAL_SPLIT, frac=config.frac)
    # val_loader = get_eval_loader(config.loader, val_data, batch_size=100)

    # At each round, evalute on all of the test data.


@timed
def prepare_data(data_loader: DataLoader, embed_input: bool = True) -> Tuple[Any]:
    # X = [[0.0, 0.0], [1.0, 1.0]]
    # y = [0, 1]
    X = []
    Y = []
    X_metadata = []
    for x, y_true, metadata in data_loader:
        # Use "extend" because the data loader returns batch_size points at a time.
        X.extend(x)
        Y.extend(y_true)
        X_metadata.extend(metadata)

    if embed_input:
        X = embed_text(X)

    logger.info(f"Prepared {len(X)} examples.")
    return X, Y, X_metadata


def embed_text(X: List[str]) -> List[List[float]]:
    embedded_sentences = SBERT_MODEL.encode(X)
    return embedded_sentences


def calculate_accuracy():
    pass


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset", choices=["amazon", "civilcomments"], required=True
    )
    parser.add_argument(
        "-f",
        "--frac",
        type=float,
        default=1.0,
        help="Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument(
        "-r",
        "--root-dir",
        default="data",
        help="The directory where data can be found.",
    )

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    main()
