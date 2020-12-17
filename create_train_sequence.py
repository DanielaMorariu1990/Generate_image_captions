"""
Creates a generator function, which generates the input for the model.
"""

import numpy as np
import pickle

from tensorflow import keras


def create_trianing_data(captions, images, tokenizer, max_caption_length, vocab_len, photos_per_batch):

    X1, X2, y = list(), list(), list()
    n = 0

    # loop through every image
    while True:
        for key, cap in captions.items():
            n += 1
            # retrieve the photo feature
            image = images[key].reshape(-1)

            for c in cap:
                # encode the sequence
                sequence = [tokenizer.word_index[word] for word in c.split(
                    ' ') if word in list(tokenizer.word_index.keys())]

                # split one sequence into multiple X, y pairs

                for i in range(1, len(sequence)):
                    # creating input, output
                    inp, out = sequence[:i], sequence[i]
                    # padding input
                    input_seq = keras.preprocessing.sequence.pad_sequences(
                        [inp], maxlen=max_caption_length)[0]
                    # encode output sequence
                    output_seq = keras.utils.to_categorical(
                        [out], num_classes=vocab_len)[0]
                    # store
                    X1.append(image)
                    X2.append(input_seq)
                    y.append(output_seq)

            # yield the batch data
            if n == photos_per_batch:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0


if __name__ == "__main__":
    from tokenize_words import max_seq_lenght, load_clean_descriptions, load_photos
    with open('./model_files/tokenizer.pkl', 'rb') as f:
        tok = pickle.load(f)

    with open('./model_files/features2.pkl', 'rb') as f:
        img = pickle.load(f)

    filename = './Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_photos(filename)
    train_descriptions = load_clean_descriptions(
        './model_files/description.json', train)
    max_length = max_seq_lenght(train_descriptions)
    vocab_size = len(tok.word_index) + 1
    train_data = create_trianing_data(train_descriptions, img,
                                      tok, max_length, vocab_size, 1)
