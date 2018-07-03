from __future__ import print_function

import os

from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding, Concatenate
from keras.models import Model

from load_glove_embeddings import load_glove_embeddings

DATA_BASE_DIR = ''

EMBEDDING_FILE_PATH = os.path.join(DATA_BASE_DIR, 'glove/glove.6B.50d.txt')
MAX_NUM_WORDS = 400001
EMBEDDING_DIM = 50
word2index, embedding_matrix = "", ""

MAX_SEQUENCE_LENGTH = 100

NUMBER_OF_FILTERS = [200]
KERNEL_SIZE = [4]
HIDDEN_LAYER = 300
SIMILARITY_LAYER = 20


def get_emdedding_layer():
    return Embedding(MAX_NUM_WORDS,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)


def embedded_cnn(x):
    # Embedding Layer
    emdded = get_emdedding_layer()(x)

    # 1D Conv Layers
    # For single kind of convolution or filter-size(n-gram)
    # x = Conv1D(filters=NUMBER_OF_FILTERS,
    #            kernel_size=KERNEL_SIZE,
    #            padding='valid',
    #            activation='relu',
    #            strides=1)(x)

    # 1D Conv Layer with multiple possible kernel sizes
    conv_layers = []
    for n_gram, hidden_units in zip(KERNEL_SIZE, NUMBER_OF_FILTERS):
        # 1D Conv with kernel size of n_gram(and with hidden_units of those filters)
        conv_layer = Conv1D(filters=hidden_units,
                            kernel_size=n_gram,
                            padding='valid',
                            activation='relu')(emdded)

        conv_layer = GlobalMaxPooling1D()(conv_layer)
        conv_layers.append(conv_layer)

    if len(conv_layers) == 1:
        return conv_layers[0]
    else:
        # Concatenates conv layers with  different filter sizes
        all_conv_layers_merged = Concatenate()(conv_layers)
        return all_conv_layers_merged


def get_model():
    # So that we load word embedding only once
    print("Loading word embedding")
    global word2index
    global embedding_matrix
    word2index, embedding_matrix = load_glove_embeddings(fp=EMBEDDING_FILE_PATH, embedding_dim=EMBEDDING_DIM)
    print("Vocab Size: %d" % len(word2index.keys()))
    print("Done loading word embedding")

    # For first sentence
    x_1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    f = embedded_cnn(x_1)  # weight x_f here

    # For second sentence
    x_2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    s = embedded_cnn(x_2)  # weight x_s here

    # For third sentence
    x_3 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    t = embedded_cnn(x_3)  # weight x_t here

    # Similarity between first and second sentence
    # NOTE: I've used a dense layer instead of similarity matrix to get the similarity score
    fs = Dense(units=SIMILARITY_LAYER,
               activation='relu')(Concatenate()([f, s]))  # M_fs here

    # Similarity between second and third sentence
    # NOTE: I've used a dense layer instead of similarity matrix to get the similarity score
    st = Dense(units=SIMILARITY_LAYER,
               activation='relu')(Concatenate()([s, t]))  # M_st here

    # makes a flat layer/single layer concatenated with....
    join_layer = Concatenate()([fs, f, s, t, st])

    hidden = Dense(units=HIDDEN_LAYER,
                   activation='relu')(join_layer)
    hidden = Dropout(0.2)(hidden)

    is_coherent = Dense(units=1,
                        activation='sigmoid')(hidden)

    model = Model(inputs=[x_1, x_2, x_3],
                  outputs=is_coherent)

    return model


if __name__ == '__main__':
    get_model().summary()
