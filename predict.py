import re

from keras.preprocessing.sequence import pad_sequences

from model import *


class CoherenceModel(object):
    def __init__(self, weight_file):

        print("Stating to load model config")
        self.model = get_model()
        print("Model config loaded")

        print("Starting to load model weights")
        self.model.load_weights(weight_file)
        print("Model weight loaded")

    @staticmethod
    def _process_line(line):
        splits = line.split(' ')
        res = []
        for element in splits:
            res.extend([x for x in re.split('(\W+)', element) if (len(x) != 0)])

        full_num = []
        for word in res:
            try:
                full_num.append(word2index[word])
            except:
                pass
        return pad_sequences([full_num], maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post', value=-1)

    def predict(self, first_sentence, second_sentence, third_sentence):
        x = [CoherenceModel._process_line(first_sentence),
             CoherenceModel._process_line(second_sentence),
             CoherenceModel._process_line(third_sentence)]

        print("Starting to predict")
        value = self.model.predict(x)
        return value[0]


if __name__ == '__main__':
    # TODO: ENTER MODEL WEIGHT FILE NAME HERE
    obj = CoherenceModel(weight_file="trained_models/<>.h5")

    # TODO: ENTER LINES HERE
    # first line
    f = ""
    # second line
    s = ""
    # third line
    t = ""

    prob = obj.predict(f, s, t)

    print("Prediction: %f", prob)
