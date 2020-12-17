"""Creates model using Keras:
input layer 1: image features, followed by 4 dense layers and 1 Dropout layer
input layer 2: text (captions), embedding layer(calibrated via word2vec), 
followed by 4 LSTM layers.
These 2 models are added then togther, followed by 2 dense layers and one
output layer.
"""

from tensorflow import keras


def create_model_img(max_caption_length, vocab_length, embedding_vectors):

    # sub network for handling the image feature part
    input_layer1 = keras.Input(shape=(2048,))
    feature1 = keras.layers.Dropout(0.2)(input_layer1)
    feature2 = keras.layers.Dense(
        max_caption_length*4, activation='relu')(feature1)
    feature3 = keras.layers.Dense(
        max_caption_length*4, activation='relu')(feature2)
    feature4 = keras.layers.Dense(
        max_caption_length*4, activation='relu')(feature3)
    feature5 = keras.layers.Dense(
        max_caption_length*4, activation='relu')(feature4)

    # sub network for handling the text generation part
    input_layer2 = keras.Input(shape=(max_caption_length,))
    cap_layer1 = keras.layers.Embedding(
        vocab_length, 300, input_length=max_caption_length,
        weights=[embedding_vectors], trainable=False)(input_layer2)
    cap_layer2 = keras.layers.Dropout(0.2)(cap_layer1)
    cap_layer3 = keras.layers.LSTM(
        max_caption_length*4, activation='relu', return_sequences=True)(cap_layer2)
    cap_layer4 = keras.layers.LSTM(
        max_caption_length*4, activation='relu', return_sequences=True)(cap_layer3)
    cap_layer5 = keras.layers.LSTM(
        max_caption_length*4, activation='relu', return_sequences=True)(cap_layer4)
    cap_layer6 = keras.layers.LSTM(
        max_caption_length*4, activation='relu')(cap_layer5)

    # merging the two sub network
    decoder1 = keras.layers.Add()([feature5, cap_layer6])
    decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
    decoder3 = keras.layers.Dense(256, activation='relu')(decoder2)

    # output is the next word in sequence
    output_layer = keras.layers.Dense(
        vocab_length, activation='softmax')(decoder3)
    model = keras.models.Model(
        inputs=[input_layer1, input_layer2], outputs=output_layer)

    model.summary()

    return model
