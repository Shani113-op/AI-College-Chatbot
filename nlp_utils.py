# nlp_utils.py
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required corpora (run once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = (text or "").lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
    return tokens

def bag_of_words(tokens, vocab):
    return {f"contains({w})": (w in tokens) for w in vocab}
