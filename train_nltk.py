# train_nltk.py
import json
import os
import pickle
import glob
import nltk
from nlp_utils import preprocess, bag_of_words

# Paths
INTENTS_DIR = "intents"           # folder containing multiple JSON files
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

# Load all intents JSON files
intents = {"intents": []}
for file in glob.glob(os.path.join(INTENTS_DIR, "*.json")):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        intents["intents"].extend(data.get("intents", []))

# Prepare training examples
training_examples = []
for item in intents["intents"]:
    tag = item["tag"]
    for p in item["patterns"]:
        training_examples.append((p, tag))

# Build vocabulary
all_words = set()
for sentence, _ in training_examples:
    toks = preprocess(sentence)
    all_words.update(toks)
vocab = sorted(all_words)

# Build feature set for NaiveBayesClassifier
train_set = []
for sentence, tag in training_examples:
    toks = preprocess(sentence)
    feats = bag_of_words(toks, vocab)
    train_set.append((feats, tag))

print(f"Training on {len(train_set)} examples, vocab size {len(vocab)}")

# Train classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump((classifier, vocab), f)

print("✅ Model saved to", MODEL_PATH)
classifier.show_most_informative_features(10)
