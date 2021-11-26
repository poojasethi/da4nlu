import argparse
import logging
import os
from re import I
import numpy as np
import os
import sys


from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader, DataLoader
from typing import Any, Tuple, List
from sentence_transformers import SentenceTransformer
from utils import timed
from joblib import dump, load


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
LOADER = "standard"

LOSS = "hinge"
PENALTY = "l2"
MAX_ITER = 5

def main():
    config = get_config()
    dataset = get_dataset(dataset=config.dataset, download=False)

    # Prepare training data
    train_data = dataset.get_subset(TRAIN_SPLIT, frac=config.frac)
    train_loader = get_train_loader(LOADER, train_data, batch_size=config.batch_size)
    X_train, Y_train, X_train_metadata = prepare_data(train_loader)

    # Prepare test data
    # test_data = dataset.get_subset(TEST_SPLIT, frac=config.frac)
    # test_loader = get_eval_loader(LOADER, test_data, batch_size=config.batch_size)
    # X_test, Y_test, X_test_metadata = prepare_data(test_loader)

    iteration = 0

    # TODO: Refactor into "train" method:
    model_path = f"{config.output_dir}/{iteration}/model_{len(X_train)}.joblib"
    results_path = f"{config.output_dir}/{iteration}/results_{len(X_train)}.txt"
    if os.path.exists(model_path):
        logger.info(f"Already found a model at {model_path}, skipping save")
    else:
        logger.info(f"Starting round {iteration} of model training...")
        clf = SGDClassifier(loss=LOSS, penalty=PENALTY, max_iter=MAX_ITER)
        clf.fit(X_train, Y_train)
        clf.predict(X_train)
        save_model_and_results(clf, model_path, results_path)

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

    logger.info(f"Prepared {len(X)} examples")
    return X, Y, X_metadata


def embed_text(X: List[str]) -> List[List[float]]:
    embedded_sentences = SBERT_MODEL.encode(X)
    return embedded_sentences


def save_model_and_results(
    clf: SGDClassifier, model_path: str, results_path: str
) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    logging.info(f"Saving model to {model_path}")
    dump(clf, model_path)

    calculate_and_save_accuracy(clf, results_path)


def calculate_and_save_accuracy(clf: SGDClassifier, results_path: str):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    logging.info(f"Saving model results to {results_path}")

    pass


def load_model(model_path: str) -> SGDClassifier:
    return load(model_path)


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
        "-o",
        "--output-dir",
        default="results",
        help="The directory to output model results to.",
    )

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    main()
