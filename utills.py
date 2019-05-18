import json
from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def pad_all(dict, maxlen):
    """

    :param dict:
    :return: 0 -> [   0 -> first sentence
                     1 -> second sentence
                     2 -> third sentence
                 ],
             1 -> label
    """
    x_1 = dict['firsts']
    x_2 = dict['seconds']
    x_3 = dict['thirds']

    x_1 = pad_sequences(x_1, maxlen=maxlen, padding='post', truncating='post', value=0)
    x_2 = pad_sequences(x_2, maxlen=maxlen, padding='post', truncating='post', value=0)
    x_3 = pad_sequences(x_3, maxlen=maxlen, padding='post', truncating='post', value=0)

    return np.array([[x_1, x_2, x_3], dict['labels']])


def load_cui_dataset(path, maxlen):
    with open(path + "train.json") as f:
        print("Loading train")
        train = json.load(f)
        print("Done loading train. \n Padding train")
        train = pad_all(train, maxlen)
        print("Done padding train")

    with open(path + "test.json") as f:
        print("Loading test")
        test = json.load(f)
        print("Done loading test. \n Padding test")
        test = pad_all(test, maxlen)
        print("Done padding test")

    with open(path + "dev.json") as f:
        print("Loading dev")
        dev = json.load(f)
        print("Done loading dev. \n Padding dev")
        dev = pad_all(dev, maxlen)
        print("Done padding dev")

    return train, dev, test


from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}
