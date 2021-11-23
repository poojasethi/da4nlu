from transformers import pipeline
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader


dataset = get_dataset(dataset='civilcomments', download=False)



# Get model predictions
classifier = pipeline('sentiment-analysis')
prediction = classifier("Wow, using HuggingFace is easy")
print(prediction)
