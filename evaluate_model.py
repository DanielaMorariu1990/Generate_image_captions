"""
Function that returns actual descriptions, predicted descriptions and image names
"""

from tensorflow import keras
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


if __name__ == "__main__":

    image_cap = image_cap = keras.models.load_model(
        "./model_files/Image_caption_v1.h5")

    with open('./model_files/description.json', 'rb') as f:
        desc = json.load(f)

    with open('./model_files/tokenizer.pkl', 'rb') as f:
        tok = pickle.load(f)

    with open('./model_files/features2.pkl', 'rb') as f:
        img = pickle.load(f)

    filename = './Flickr8k_text/Flickr_8k.testImages.txt'
    test = load_photos(filename)
    test_descriptions = load_clean_descriptions(
        './model_files/description.json', test)

    keys, actual, pred = model_evaluation(model=image_cap, feature_vector=img,
                                          max_caption_length=34, tokenizer=tok, description=test_descriptions)

    with open("./model_files/actual2.pkl", "wb") as fp:  # Pickling
        pickle.dump(actual, fp)

    with open("./model_files/pred2.pkl", "wb") as fp:  # Pickling
        pickle.dump(pred, fp)

    with open("./model_files/keys.pkl", "wb") as fp:  # Pickling
        pickle.dump(keys, fp)
