import pickle
import numpy as np
from tensorflow import keras

from generate_sequence import create_trianing_data
from tokenize_words import max_seq_lenght, load_clean_descriptions, load_photos
from create_model import create_model_img

# Load needed data
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('features2.pkl', 'rb') as f:
    img_features = pickle.load(f)

with open('embedding_vectors.pkl', 'rb') as f:
    embedding_vectors = pickle.load(f)

# get training data

filename = './Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_photos(filename)
train_descriptions = load_clean_descriptions('./description.json', train)

max_length_caption = max_seq_lenght(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1

train_data = create_trianing_data(
    train_descriptions, img_features, tokenizer, max_length_caption, vocab_size, 32)


# # validation data set
# filename_v = 'Flickr8k_text/Flickr_8k.devImages.txt'
# validation = load_photos(filename_v)
# valid_descriptions = load_clean_descriptions('./description.json', validation)
# valid_data = create_trianing_data(
#     valid_descriptions, img_features, tokenizer, max_length_caption, vocab_size, 32)


# initialize model
model_cap = create_model_img(max_length_caption, vocab_size, embedding_vectors)

steps_per_epochs = len(train_descriptions)//32

# compile model
model_cap.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=["accuracy"])


callback = keras.callbacks.EarlyStopping(
    monitor='loss', patience=3)

history = model_cap.fit_generator(generator=train_data, epochs=40,
                                  steps_per_epoch=steps_per_epochs,
                                  callbacks=[callback])

model_cap.save("Image_caption_v1.h5")
