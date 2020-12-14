import json
import pandas as pd
import os
import pickle
import re
import string
import pickle
import _pickle as pickle
from collections import defaultdict


def read_captions(path):
    caption_dict = {}
    with open(path, 'r') as f:
        for line in f:
            splitLine = line.split('\t')
            caption_dict[str(splitLine[0])] = splitLine[1:]

    return caption_dict


def clean_description(dict_input):
    keys = []
    description = []
    description_final = defaultdict(list)
    for key, item in dict_input.items():
        key = key[:-2]
        desc = item[0].split()
        desc = [i.lower() for i in desc]
        desc = [i.translate(str.maketrans('', '', string.punctuation))
                for i in desc]
        desc = [i.rstrip() for i in desc]
        desc = [i.lstrip() for i in desc]
        desc = [i for i in desc if len(i) > 1]
        desc = ' '.join([i for i in desc if not i.isdigit()])
        keys.append(key)
        description.append(desc)
    for t in zip(keys, description):
        description_final[t[0]].append(t[1])

    return description_final


def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab


if __name__ == "__main__":
    captions_path = './Flickr8k_text/Flickr8k.token.txt'
    caption_dict = read_captions(captions_path)
    description_clean = clean_description(caption_dict)
    vocabulary = text_vocabulary(description_clean)
    print(len(vocabulary))

    # save file to disk
    with open('description.json', 'w') as fp:
        json.dump(description_clean, fp)

    # with open("vocabulary.txt", "wb") as fp:  # Pickling
    #     pickle.dump(vocabulary, fp)
