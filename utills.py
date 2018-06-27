import json
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

    x_1 = pad_sequences(x_1, maxlen=maxlen)
    x_2 = pad_sequences(x_2, maxlen=maxlen)
    x_3 = pad_sequences(x_3, maxlen=maxlen)

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
