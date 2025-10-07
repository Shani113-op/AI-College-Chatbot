import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_utils import preprocess, bag_of_words

def classify_intent(text, classifier, vocab):
    feats = bag_of_words(preprocess(text), vocab)
    prob_dist = classifier.prob_classify(feats)
    best = prob_dist.max()
    conf = prob_dist.prob(best)
    return best, conf

def get_default_response(tag, intents):
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "🤔 I didn’t understand. Try asking about college info, fees, placements, cutoffs, or deadlines."
