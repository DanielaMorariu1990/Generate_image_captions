from tensorflow import keras
import nltk
import json
import pickle

from generate_captions import generate_captions
from tokenize_words import load_clean_descriptions, load_photos


def model_evaluation(model, feature_vector, max_caption_length, tokenizer,
                     description):

    actual = list()
    predicted = list()
    keys = list()
    for img, desc in description.items():
        pred = generate_captions(model,
                                 img, tokenizer, max_caption_length, feature_vector)
        act = [line.split() for line in desc]
        act = act[1:-1]
        predicted.append(pred.split())
        actual.append(act)
        keys.append(img)
    return keys, actual, predicted


image_cap = image_cap = keras.models.load_model("Image_caption_v1.h5")

with open('description.json', 'rb') as f:
    desc = json.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

with open('features2.pkl', 'rb') as f:
    img = pickle.load(f)

filename = './Flickr8k_text/Flickr_8k.testImages.txt'
test = load_photos(filename)
test_descriptions = load_clean_descriptions('./description.json', test)


keys, actual, pred = model_evaluation(model=image_cap, feature_vector=img,
                                      max_caption_length=34, tokenizer=tok, description=test_descriptions)

with open("actual2.pkl", "wb") as fp:  # Pickling
    pickle.dump(actual, fp)

with open("pred2.pkl", "wb") as fp:  # Pickling
    pickle.dump(pred, fp)

with open("keys.pkl", "wb") as fp:  # Pickling
    pickle.dump(keys, fp)
