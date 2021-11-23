from transformers import pipeline

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader

DATASET = "civilcomments"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

STANDARD_LOADER = "standard"

dataset = get_dataset(dataset=DATASET, download=False)

train_data = dataset.get_subset(TRAIN_SPLIT, frac=0.01)
train_loader = get_train_loader(STANDARD_LOADER, train_data, batch_size=16)
for x, y_true, metadata in train_loader:
    print(f"{x}, {y_true}, {metadata}")
    # TODO(pooja): Train the model!

val_data = dataset.get_subset(VAL_SPLIT, frac=1.0)
val_loader = get_eval_loader(STANDARD_LOADER, val_data, batch_size=100)
# TODO(pooja): Write sampling strategies.

# Get model predictions
classifier = pipeline("sentiment-analysis")
prediction = classifier("Wow, using HuggingFace is easy")
print(prediction)
