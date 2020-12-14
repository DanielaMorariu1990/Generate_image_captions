from tensorflow import keras
from pickle import dump
import json
import pickle

from clean_descriptions import read_captions


# load a pre-defined list of photo tags

def load_photos(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    file2 = text
    photos = file2.split("\n")[:-1]
    return photos

# load clean descriptions


def load_clean_descriptions(path, dataset):
    # load document
    with open(path) as f:
        doc = json.load(f)
    descriptions = dict()
    for key, value in doc.items():

        image_id, image_desc = key, value
        # skip images not in the set
        for vals in image_desc:
            if image_id in dataset:
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # wrap description in tokens
                desc = 'startseq ' + str(vals) + ' endseq'
                # store
                descriptions[image_id].append(desc)
    return descriptions

# covert a dictionary of clean descriptions to a list of descriptions


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions


def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_seq_lenght(description):
    lines = to_lines(description)
    maximum = max(len(d.split()) for d in lines)
    return maximum


if __name__ == "__main__":

    # load training dataset (6K)
    filename = './Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_photos(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    train_descriptions = load_clean_descriptions('./description.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    # save the tokenizer
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    # max length of the sequence
    print("max length:")
    print(max_seq_lenght(train_descriptions))
