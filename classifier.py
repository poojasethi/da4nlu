from transformers import pipeline

classifier = pipeline('sentiment-analysis')
prediction = classifier("Wow, using HuggingFace is easy")
print(prediction)
