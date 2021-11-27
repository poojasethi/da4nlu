import argparse
import logging
import os
import sys
from enum import Enum
from re import I
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from utils import timed
from wilds import get_dataset
from wilds.common.data_loaders import DataLoader, get_eval_loader, get_train_loader
from wilds.datasets.wilds_dataset import WILDSDataset

import matplotlib.pyplot as plt
from dataclasses import dataclass

"""
This script trains a binary classification model on the Civil Comments WILDS dataset.
It trains an initial "seed" model, then tries to improve the model in an online setting,
by either sampling using self-learning, uncertainty-based active learning, or one of 
the other sampling strategies defined here.
"""

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
    LOW_CONFIDENCE_INVERSE_PREDICTION = "low_confidence_inverse_prediction"

SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
LOADER = "standard"


@dataclass
class Hyperparameters:
    loss = "hinge"
    max_training_iter = 10
    penalty = "elasticnet"


X_Y_Metadata_Sentences = Tuple[
    List[List[float]], List[int], List[List[int]], Optional[List[str]]
]

TRAIN_RESULTS = {s.value: {} for s in SamplingStrategy}
TEST_RESULTS = {s.value: {} for s in SamplingStrategy}


@timed
def main():
    config = get_config()
    dataset = get_dataset(dataset=config.dataset, download=False)

    logger.info("Preparing training data...")
    train_data = dataset.get_subset(TRAIN_SPLIT, frac=config.frac)
    train_loader = get_train_loader(LOADER, train_data, batch_size=config.batch_size)
    X_train, Y_train, X_train_metadata, X_train_sentences = prepare_data(train_loader)

    logger.info("Preparing test data...")
    test_data = dataset.get_subset(TEST_SPLIT, frac=config.frac)
    test_loader = get_eval_loader(LOADER, test_data, batch_size=config.batch_size)
    X_test, Y_test, X_test_metadata, X_test_sentences = prepare_data(test_loader)
    test_data = (X_test, Y_test, X_test_metadata, X_test_sentences)

    # Initialize the "seed" model that we will use for our sampling experiments.
    iteration = 0
    model_path = f"{config.output_dir}/{iteration}/model.joblib"
    results_path = f"{config.output_dir}/{iteration}/results.txt"
    if os.path.exists(model_path):
        logger.info(
            f"Already found a model trained with {len(X_train)} samples at {model_path}, skipping training!"
        )
    else:
        logger.info(f"Starting round {iteration} of model training...")
        classes = np.unique(Y_train)
        balanced_class_weights = compute_class_weight(
            "balanced", classes=classes, y=Y_train
        )
        class_weight = dict(zip(classes, balanced_class_weights))
        logging.info(f"Computed balanced class weights: {class_weight}...")
        clf = SGDClassifier(
            loss=Hyperparameters.loss,
            max_iter=Hyperparameters.max_training_iter,
            penalty=Hyperparameters.penalty,
            class_weight=class_weight,
        )
        clf.fit(X_train, Y_train)
        clf.predict(X_train)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(clf, model_path)
        results_dict = save_results(
            clf,
            results_path,
            {
                TRAIN_SPLIT: (X_train, Y_train, X_train_metadata, X_train_sentences),
                TEST_SPLIT: test_data,
            },
            dataset,
        )
        for name, results in results_dict.items():
            if name == TRAIN_SPLIT:
                for _, iterations in TRAIN_RESULTS.items():
                    iterations[0] = results
            elif name == TEST_SPLIT:
                for _, iterations in TEST_RESULTS.items():
                    iterations[0] = results

    # Perform online sampling and retraining.
    sample_data_and_retrain(
        dataset,
        config.frac,
        config.sampling_frac,
        config.batch_size,
        config.output_dir,
        config.num_sampling_iter,
        test_data,
    )

    # Plot results.
    for name, results in {TRAIN_SPLIT: TRAIN_RESULTS, TEST_SPLIT: TEST_RESULTS}.items():
        plot_path = f"{config.output_dir}/{name}_results.png"
        plot_results(name, results, plot_path)


def sample_data_and_retrain(
    dataset: WILDSDataset,
    frac: float,
    sampling_frac: float,
    batch_size: int,
    output_dir: str,
    num_sampling_iter: int,
    test_data: X_Y_Metadata_Sentences,
):
    for i in range(1, 1 + num_sampling_iter):
        logger.info(f"Starting round {i} of model training...")

        # The validation set is our candidate pool of unlabeled data for sampling.
        logger.info("Preparing validation data (candidate pool) for sampling...")
        val_data = dataset.get_subset(VAL_SPLIT, frac=frac)
        val_loader = get_eval_loader(LOADER, val_data, batch_size=batch_size)
        X_val, Y_val, X_val_metadata, X_val_sentences = prepare_data(val_loader)
        candidate_pool = pd.DataFrame(
            {
                "X": X_val,
                "Y": Y_val,
                "X_metadata": X_val_metadata,
                "X_sentences": X_val_sentences,
            }
        )

        for strategy in SamplingStrategy:
            logger.info(f"Running sampling strategy {strategy}...")
            model_path = f"{output_dir}/{i}/{strategy.value}/model.joblib"
            results_path = f"{output_dir}/{i}/{strategy.value}/results.txt"
            data_path = f"{output_dir}/{i}/{strategy.value}/data.csv"

            if os.path.exists(model_path) and os.path.exists(results_path):
                logger.info(
                    f"Already found a model trained at {model_path} and results saved at {results_path} -- skipping training!"
                )
            else:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                previous_model_path = (
                    f"{output_dir}/{i - 1}/model.joblib"
                    if i == 1
                    else f"{output_dir}/{i - 1}/{strategy.value}/model.joblib"
                )

                clf, sampled_data = run_sampling_strategy(
                    strategy,
                    candidate_pool,
                    sampling_frac,
                    previous_model_path,
                    data_path,
                )
                save_model(clf, model_path)
                results_dict = save_results(
                    clf,
                    results_path,
                    {TRAIN_SPLIT: sampled_data, TEST_SPLIT: test_data},
                    dataset,
                )
                for name, results in results_dict.items():
                    if name == TRAIN_SPLIT:
                        TRAIN_RESULTS[strategy.value][i] = results
                    elif name == TEST_SPLIT:
                        TEST_RESULTS[strategy.value][i] = results


@timed
def run_sampling_strategy(
    strategy: SamplingStrategy,
    candidate_pool: pd.DataFrame,
    sampling_frac: float,
    previous_model_path: str,
    data_path: str,
) -> Tuple[SGDClassifier, X_Y_Metadata_Sentences]:
    logger.info(f"Loading model from {previous_model_path}...")
    clf = load(previous_model_path)
    sampled_data = sample_data(candidate_pool, strategy, sampling_frac, clf, data_path)

    X_sampled, Y_sampled, X_sampled_metadata, X_sentences = (
        sampled_data["X"],
        sampled_data["Y"],
        sampled_data["X_metadata"],
        sampled_data["X_sentences"],
    )
    X_sampled, Y_sampled, X_sampled_metadata = (
        X_sampled.tolist(),
        Y_sampled.tolist(),
        X_sampled_metadata.tolist(),
    )
    clf.partial_fit(X_sampled, Y_sampled)
    return clf, (X_sampled, Y_sampled, X_sampled_metadata, X_sentences)


@timed
def sample_data(
    candidate_pool: pd.DataFrame,
    strategy: SamplingStrategy,
    sampling_frac: float,
    clf: SGDClassifier,
    data_path: str,
    write_data: bool = True,
) -> pd.DataFrame:
    if strategy == SamplingStrategy.RANDOM:
        return candidate_pool.sample(frac=sampling_frac)
    else:
        candidate_info = candidate_pool.copy(deep=True)
        X = candidate_pool["X"].tolist()
        candidate_info["Y_predict"] = clf.predict(X)
        candidate_info["Y_decision"] = clf.decision_function(X)
        candidate_info["Y_confidence"] = abs(candidate_info["Y_decision"])

        # Sort the sampled data by confidence in descending order (high confidence is at the top).
        candidate_info.sort_values("Y_confidence", ascending=False, inplace=True)

        num_to_sample = int(len(candidate_pool) * sampling_frac)
        if strategy == SamplingStrategy.HYBRID:
            head = candidate_info.head(n=(num_to_sample // 2)).copy(deep=True)
            tail = candidate_info.tail(n=(num_to_sample // 2)).copy(deep=True)
            head["Y"] = head["Y_predict"]
            sampled_data = pd.concat([head, tail])
        else:
            # Return sampled data.
            if is_high_confidence_strategy(strategy):
                sampled_data = candidate_info.head(n=num_to_sample).copy(deep=True)
            else:
                sampled_data = candidate_info.tail(n=num_to_sample).copy(deep=True)

            # For self-learning, replace the gold label with the model prediction.
            if is_self_learning_strategy(strategy):
                if strategy == SamplingStrategy.LOW_CONFIDENCE_INVERSE_PREDICTION:
                    # For low confidence examples, set the gold label to the opposite 
                    # of what the model predicted. This only works in the binary 
                    # classification case of course.
                    sampled_data["Y"] = (~(sampled_data["Y_predict"].astype(bool))).astype(int)
                else:
                    sampled_data["Y"] = sampled_data["Y_predict"]

        examples = sampled_data[["X_sentences", "Y", "Y_decision", "Y_confidence"]]
        logger.info(
            f"Sampled some data! Here are some examples:\n {examples.head(n=10)}"
        )
        if write_data:
            examples.to_csv(data_path)

        return sampled_data


def is_self_learning_strategy(strategy):
    return (
        strategy == SamplingStrategy.SELF_LEARNIG
        or strategy == SamplingStrategy.LOW_CONFIDENCE_USE_PREDICTION
        or strategy == SamplingStrategy.LOW_CONFIDENCE_INVERSE_PREDICTION
    )


def is_high_confidence_strategy(strategy):
    return (
        strategy == SamplingStrategy.SELF_LEARNIG
        or strategy == SamplingStrategy.HIGH_CONFIDENCE_USE_GOLD
    )


@timed
def prepare_data(
    data_loader: DataLoader, embed_input: bool = True
) -> X_Y_Metadata_Sentences:
    X_sentences = []
    Y = []
    X_metadata = []
    for x, y_true, metadata in data_loader:
        # Use "extend" because the data loader returns batch_size points at a time.
        X_sentences.extend(x)
        Y.extend(y_true)
        X_metadata.extend(metadata)

    # Conver tensors to Python lists.
    Y = [y.item() for y in Y]
    X_metadata = [m.tolist() for m in X_metadata]
    if embed_input:
        X = embed_text(X_sentences).tolist()

    logger.info(f"Prepared {len(X)} examples")
    return X, Y, X_metadata, X_sentences


def embed_text(X: List[str]) -> List[List[float]]:
    embedded_sentences = SBERT_MODEL.encode(X)
    return embedded_sentences


def save_model(
    clf: SGDClassifier,
    model_path: str,
) -> None:
    logging.info(f"Saving model path to: {model_path}")
    dump(clf, model_path)


def save_results(
    clf: SGDClassifier,
    results_path: str,
    eval_sets: Dict[str, X_Y_Metadata_Sentences],
    dataset: WILDSDataset,
) -> Dict[str, Dict[str, Any]]:
    results_dict = {}
    for name, data in eval_sets.items():
        results_dict[name] = calculate_and_save_accuracy(
            clf, results_path, name, data, dataset
        )
    return results_dict


def calculate_and_save_accuracy(
    clf: SGDClassifier,
    results_path: str,
    name: str,
    eval_data: X_Y_Metadata_Sentences,
    dataset: WILDSDataset,
) -> Dict[str, Any]:
    X_eval, Y_eval, X_eval_metadata, _ = eval_data
    # Note: Because we do not necessarily need a calibrated probability but just a
    # confidence score, calling predict here should be ok.
    Y_predict_np_array = clf.predict(X_eval)

    Y_predict_tensor = torch.from_numpy(Y_predict_np_array)
    Y_eval_tensor = torch.tensor(Y_eval)
    X_eval_metadata_tensor = torch.stack([torch.tensor(x) for x in X_eval_metadata])

    results, results_str = dataset.eval(
        Y_predict_tensor, Y_eval_tensor, X_eval_metadata_tensor
    )
    logger.info(results_str)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    logging.info(f"Saving results to: {results_path}")
    with open(results_path, mode="a") as fh:
        fh.write(f"Results for: {name}\n")
        fh.write(results_str)
        fh.write(f"{'*' * 30}\n")

    return results


def plot_results(
    name: str, results: Dict[str, Dict[int, Dict[str, Any]]], plot_path: str
):
    for strategy, iterations in results.items():
        test_accuracies = []
        for i, i_results in iterations.items():
            test_accuracies.append(i_results["acc_avg"])

        plt.plot(iterations.keys(), test_accuracies, label=strategy)

    plt.title(f"{name.capitalize()} Accuracy vs. Iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Average Accuracy")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()


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
    parser.add_argument(
        "-n",
        "--num-sampling-iter",
        type=int,
        default=5,
        help="The number of sampling iterations to perform.",
    )

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    main()
