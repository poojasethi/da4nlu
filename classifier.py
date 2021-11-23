from transformers import pipeline
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

DATASET = 'civilcomments'

TRAIN_SPLIT = 'train'
EVAL_SPLIT = 'eval'

STANDARD_LOADER = 'standard'

dataset = get_dataset(dataset=DATASET, download=False)

train_data = dataset.get_subset(TRAIN_SPLIT, frac=0.01)
train_loader = get_train_loader(STANDARD_LOADER, train_data, batch_size=16)
for x, y_true, metadata in train_loader:
    print(f"{x}, {y_true}, {metadata}")


eval_data = dataset.get_subset(EVAL_SPLIT, frac=1.0)
eval_loader = get_eval_loader(STANDARD_LOADER, eval_data, batch_size=100)

# Get model predictions
classifier = pipeline('sentiment-analysis')
prediction = classifier("Wow, using HuggingFace is easy")
print(prediction)
