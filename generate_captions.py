"""Generate new captions based on the trained model.
"""

from tensorflow import keras
import pickle
import numpy as np


def generate_captions(model, image, tokenizer, max_caption_length, feature_vector):
    """Generates captions for one image at a time. 
    Assumes that the features are already extracted.
    Stops when reaching endseq.
    ### Parametrs:
        model: NN model, pre-trained
        image=image name (.jpg)
        tokenizer= tokenized words
        max_caption_length: maximum length of caption
        feature_vector: vector extracted from Inception model(or other model)
        containg image features

    ###Returns:
        new generated caption, description, for given photo.
    """
    # extract image features
    image_vec = feature_vector[image]

    # input is startseq
    input_text = 'startseq'
    i = 0
    # keep generating words till we have encountered <end>
    while i <= max_caption_length:
        seq = [tokenizer.word_index[w]
               for w in input_text.split() if w in list(tokenizer.word_index.keys())]
        seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_caption_length)
        prediction = model.predict([image_vec, seq], verbose=0)
        prediction = np.argmax(prediction)
        word = tokenizer.index_word[prediction]
        input_text += ' ' + word
        if word == 'endseq':
            break
        i += 1

    # remove 'startseq' and endseq from output and return string
    output = input_text.split()
    output = output[1:-1]
    output = ' '.join(output)
    return output


if __name__ == "__main__":

    image_cap = keras.models.load_model("./model_files/Image_caption_v1.h5")
    with open('./model_files/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('./model_files/features2.pkl', 'rb') as f:
        img_features = pickle.load(f)
    image = '1584315962_5b0b45d02d.jpg'
    generate_captions(model=image_cap, image=image, tokenizer=tokenizer,
                      max_caption_length=34, feature_vector=img_features)
