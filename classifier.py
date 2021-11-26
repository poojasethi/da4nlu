import argparse
import logging
import os
import sys
from re import I
from enum import Enum
from typing import Any, List, Tuple, Dict
from numpy.random.mtrand import sample
import torch
import pandas as pd

import numpy as np
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from utils import timed
from wilds import get_dataset
from wilds.common.data_loaders import DataLoader, get_eval_loader, get_train_loader
from wilds.datasets.wilds_dataset import WILDSDataset

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class SamplingStrategy(Enum):
    RANDOM = "random"
    SELF_LEARNIG = "self_learning"  # High confidence, use prediction.
    ACTIVE_LEARNING = "active_learning"  # Low confidence, use gold.
    HYBRID = "hybrid"  # Hybrid of self learning and active learning.
    HIGH_CONFIDENCE_USE_GOLD = "high_confidence_use_gold"
    LOW_CONFIDENCE_USE_PREDICTION = "low_confidence_use_prediction"


SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
LOADER = "standard"

LOSS = "hinge"
PENALTY = "l2"
MAX_TRAINING_ITER = 5
NUM_SAMPLING_ITER = 5

X_Y_Metadata = Tuple[List[List[float]], List[int], List[List[int]]]


def main():
    config = get_config()
    dataset = get_dataset(dataset=config.dataset, download=False)

    logger.info("Preparing training data...")
    train_data = dataset.get_subset(TRAIN_SPLIT, frac=config.frac)
    train_loader = get_train_loader(LOADER, train_data, batch_size=config.batch_size)
    X_train, Y_train, X_train_metadata = prepare_data(train_loader)

    logger.info("Preparing test data...")
    test_data = dataset.get_subset(TEST_SPLIT, frac=config.frac)
    test_loader = get_eval_loader(LOADER, test_data, batch_size=config.batch_size)
    X_test, Y_test, X_test_metadata = prepare_data(test_loader)

    # Initialize the "seed" model that we will use for our sampling experiments.
    iteration = 0
    model_path = f"{config.output_dir}/{iteration}/model_{len(X_train)}.joblib"
    results_path = f"{config.output_dir}/{iteration}/results_{len(X_train)}.txt"
    if os.path.exists(model_path):
        logger.info(
            f"Already found a model trained with {len(X_train)} samples at {model_path}, skipping training!"
        )
    else:
        logger.info(f"Starting round {iteration} of model training...")
        clf = SGDClassifier(loss=LOSS, penalty=PENALTY, max_iter=MAX_TRAINING_ITER)
        clf.fit(X_train, Y_train)
        clf.predict(X_train)
        save_model(clf, model_path)
        save_results(
            clf, results_path, {"train": (X_train, Y_train, X_train_metadata), "test": (X_test, Y_test, X_test_metadata)}, dataset
        )

    # Perform online sampling and retraining.
    sample_data_and_retrain(
        dataset,
        config.frac,
        config.sampling_frac,
        config.batch_size,
        config.output_dir,
        (X_test, Y_test, X_test_metadata),
    )


def sample_data_and_retrain(
    dataset: WILDSDataset,
    frac: float,
    sampling_frac: float,
    batch_size: int,
    output_dir: str,
    test_data: X_Y_Metadata,
):
    for i in range(1, 1 + NUM_SAMPLING_ITER):
        logger.info(f"Starting round {i} of model training...")

        # The validation set is our candidate pool of unlabeled data for sampling.
        logger.info("Preparing validation data for sampling...")
        val_data = dataset.get_subset(VAL_SPLIT, frac=frac, shuffle=True)
        val_loader = get_eval_loader(LOADER, val_data, batch_size=batch_size)
        X_val, Y_val, X_val_metadata = prepare_data(val_loader)
        candidate_pool = pd.DataFrame(
            {"X": X_val, "Y": Y_val, "X_metadata": X_val_metadata}
        )

        for strategy in [s.value for s in SamplingStrategy]:
            logger.info(f"Running sampling strategy: {strategy}...")
            candidate_pool_size = len(candidate_pool)
            model_path = (
                f"{output_dir}/{i}/{strategy}/model_{candidate_pool_size}.joblib"
            )
            results_path = (
                f"{output_dir}/{i}/{strategy}/results_{candidate_pool_size}.txt"
            )
            if os.path.exists(model_path):
                logger.info(
                    f"Already found a model trained with {candidate_pool_size} samples at {model_path}, skipping training!"
                )
            else:
                previous_model_path = (
                    f"{output_dir}/{i - 1}/model_{candidate_pool_size}.joblib"
                    if i == 1
                    else f"{output_dir}/{i - 1}/{strategy}/model_{candidate_pool_size}.joblib"
                )

                clf, sampled_data = run_sampling_strategy(
                    strategy, candidate_pool, sampling_frac, previous_model_path
                )
                # Pass in test sets here.
                save_model(clf, model_path)
                save_results(
                    clf,
                    results_path,
                    {"train": sampled_data, "test": test_data},
                    dataset,
                )


@timed
def run_sampling_strategy(
    strategy: str,
    candidate_pool: pd.DataFrame,
    sampling_frac: float,
    previous_model_path: str,
) -> Tuple[SGDClassifier, X_Y_Metadata]:
    logger.info(f"Loading and retraining model from {previous_model_path}")
    clf = load(previous_model_path)
    sampled_data = sample_data(candidate_pool, strategy, sampling_frac, clf)

    X_sampled, Y_sampled, X_sampled_metadata = (
        sampled_data["X"],
        sampled_data["Y"],
        sampled_data["X_metadata"],
    )
    clf.partial_fit(X_sampled, Y_sampled)
    return clf, (X_sampled, Y_sampled, X_sampled_metadata)


@timed
def sample_data(
    candidate_pool: pd.DataFrame,
    strategy: str,
    sampling_frac: float,
    clf: SGDClassifier,
) -> pd.DataFrame:
    if strategy == SamplingStrategy.RANDOM:
        return candidate_pool.sample(frac=sampling_frac)
    else:
        sampled_data = candidate_pool
        sampled_data["Y_predict"] = clf.predict(candidate_pool["X"])
        sampled_data["Y_decision"] = clf.decision_function(candidate_pool["X"])
        sampled_data["Y_confidence"] = abs(sampled_data["Y_decision"])

        # Sort the sampled data by confidence in descending order (high confidence is at the top).
        sampled_data.sort_values("Y_confidence", ascending=False, inplace=True)

        num_to_sample = len(candidate_pool) * sampling_frac
        if strategy == SamplingStrategy.HYBRID:
            head = sampled_data.head(n=(num_to_sample / 2))
            tail = sampled_data.tail(n=(num_to_sample / 2))
            head["Y"] = head["Y_predict"]
            return pd.concact([head, tail], axis=1)
        else:
            # For self-learning, replace the gold label with the model prediction.
            if is_self_learning_strategy(strategy):
                sampled_data["Y"] = sampled_data["Y_predict"]

            # Return sampled data.
            if is_high_confidence_strategy(strategy):
                return sampled_data.head(n=num_to_sample)
            else:
                return sampled_data.tail(n=num_to_sample)


def is_self_learning_strategy(strategy):
    return (
        strategy == SamplingStrategy.SELF_LEARNIG
        or strategy == SamplingStrategy.LOW_CONFIDENCE_USE_PREDICTION
    )


def is_high_confidence_strategy(strategy):
    return (
        strategy == SamplingStrategy.SELF_LEARNIG
        or strategy == SamplingStrategy.HIGH_CONFIDENCE_USE_GOLD
    )


@timed
def prepare_data(data_loader: DataLoader, embed_input: bool = True) -> X_Y_Metadata:
    X = []
    Y = []
    X_metadata = []
    for x, y_true, metadata in data_loader:
        # Use "extend" because the data loader returns batch_size points at a time.
        X.extend(x)
        Y.extend(y_true)
        X_metadata.extend(metadata)

    # Extract labels from tensors
    # Note: Need to use mLabelEncoder here if we need to transform multi-class.
    Y = [y.item() for y in Y]

    if embed_input:
        X = embed_text(X)

    logger.info(f"Prepared {len(X)} examples")
    return X, Y, X_metadata


def embed_text(X: List[str]) -> List[List[float]]:
    embedded_sentences = SBERT_MODEL.encode(X)
    return embedded_sentences


def save_model(
    clf: SGDClassifier,
    model_path: str,
) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logging.info(f"Saving model path to: {model_path}")
    dump(clf, model_path)


def save_results(
    clf: SGDClassifier,
    results_path: str,
    eval_sets: Dict[str, X_Y_Metadata],
    dataset: WILDSDataset,
) -> None:
    for name, data in eval_sets.items():
        calculate_and_save_accuracy(clf, results_path, name, data, dataset)


def calculate_and_save_accuracy(
    clf: SGDClassifier,
    results_path: str,
    name: str,
    eval_data: X_Y_Metadata,
    dataset: WILDSDataset,
) -> None:
    X_eval, Y_eval, X_eval_metadata = eval_data
    # Note: Because we do not necessarily need a calibrated probability but just a
    # confidence score, calling predict here should be ok.
    Y_predict_np_array = clf.predict(X_eval)

    Y_predict_tensor = torch.from_numpy(Y_predict_np_array)
    Y_eval_tensor = torch.tensor(Y_eval)
    X_eval_metadata_tensor = torch.stack(X_eval_metadata)

    _, results_str = dataset.eval(
        Y_predict_tensor, Y_eval_tensor, X_eval_metadata_tensor
    )
    logger.info(results_str)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    logging.info(f"Saving results to: {results_path}")
    with open(results_path, mode="a") as fh:
        fh.write(f"Results for: {name}")
        fh.write(results_str)
        fh.write(f"{'*' * 30}\n")


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset", choices=["amazon", "civilcomments"], required=True
    )
    parser.add_argument(
        "-f",
        "--frac",
        type=float,
        default=0.5,
        help="Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.",
    )
    parser.add_argument(
        "-s",
        "--sampling-frac",
        type=float,
        default=0.1,
        help="The fraction of data to sample from the candidate pool for re-training. If we originally set frac = 0.5 and sampling-frac = 0.1, then the overall amount of data sampled will be 0.5 * 0.1 = 0.05 of the original dataset split.",
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
