import json
import os
import re
import sys

sys.path.append('../../')

from load_glove_embeddings import load_glove_embeddings


def load_data(fname, word2index):
    lines = open(fname).readlines()
    firsts, seconds, thirds, fulls, labels = [], [], [], [], []
    count = -1
    cut = 0
    for i, line in enumerate(lines):
        line = line.strip()
        cut = line.find(' ')
        line = line[cut + 1:]
        full = line.lower().split('\t')
        splits = full[0].split(' ')
        res = []
        for element in splits:
            res.extend([x for x in re.split('(\W+)', element) if (len(x) != 0)])
        full = res[:]

        # full = re.sub(r'[^\w\s]', '', full)

        full_num = []
        for word in full:
            try:
                full_num.append(word2index[word])
            except:
                pass
        full = full_num
        fulls.append(full)
        count = count + 1

        label = 1 if fname[-2:] == '-1' else 0
        labels.append(label)

    firsts = fulls[:]
    seconds = fulls[:]
    thirds = fulls[:]

    del firsts[count]
    del firsts[count - 1]
    del seconds[count]
    del seconds[0]
    del thirds[0]
    del thirds[0]
    del labels[count]
    del labels[count - 1]

    return firsts, seconds, thirds, fulls, labels


if __name__ == '__main__':

    DATA_BASE_DIR = '../../'

    EMBEDDING_FILE_PATH = os.path.join(DATA_BASE_DIR, 'glove/glove.6B.50d.txt')
    MAX_NUM_WORDS = 400001
    EMBEDDING_DIM = 50

    MAX_SEQUENCE_LENGTH = 200
    print("Loading word embedding")
    word2index, embedding_matrix = load_glove_embeddings(fp=EMBEDDING_FILE_PATH,
                                                         embedding_dim=EMBEDDING_DIM)
    print("vocab size: %s" % len(word2index.keys()))
    print("Done loading word embedding")

    train = {
        'firsts': [],
        'seconds': [],
        'thirds': [],
        'fulls': [],
        'labels': []
    }
    dev = {
        'firsts': [],
        'seconds': [],
        'thirds': [],
        'fulls': [],
        'labels': []
    }
    test = {
        'firsts': [],
        'seconds': [],
        'thirds': [],
        'fulls': [],
        'labels': []
    }

    #  200
    rootdir = 'data'
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if not filename[0] == '.':  # to avoid '.DS_Store'
                filename = os.path.join(parent, filename)
                print(filename)
                firsts, seconds, thirds, fulls, labels = load_data(filename, word2index)
                print(labels)
                train['firsts'].extend(firsts)
                train['seconds'].extend(seconds)
                train['thirds'].extend(thirds)
                train['fulls'].extend(fulls)
                train['labels'].extend(labels)

    rootdir_2 = 'data2/train-perm'
    for parent, dirnames, filenames in os.walk(rootdir_2):
        for filename in filenames:
            if not filename[-1] == 't':
                if not filename[0] == '.':
                    filename = os.path.join(parent, filename)
                    # print(filename)
                    firsts, seconds, thirds, fulls, labels = load_data(filename, word2index)
                    # print(labels)
                    train['firsts'].extend(firsts)
                    train['seconds'].extend(seconds)
                    train['thirds'].extend(thirds)
                    train['fulls'].extend(fulls)
                    train['labels'].extend(labels)
    print("Train : size is %d" % len(train['firsts']))

    with open('processed/train.json', 'w') as f:
        json.dump(train, f, ensure_ascii=False)
    del train

    rootdir_3 = 'data2/dev-perm'
    for parent, dirnames, filenames in os.walk(rootdir_3):
        for filename in filenames:
            if not filename[-1] == 't':
                if not filename[0] == '.':
                    filename = os.path.join(parent, filename)
                    # print(filename)
                    firsts, seconds, thirds, fulls, labels = load_data(filename, word2index)
                    # print(labels)
                    dev['firsts'].extend(firsts)
                    dev['seconds'].extend(seconds)
                    dev['thirds'].extend(thirds)
                    dev['fulls'].extend(fulls)
                    dev['labels'].extend(labels)

    print("Dev : size is %d" % len(dev['firsts']))

    with open('processed/dev.json', 'w') as f:
        json.dump(dev, f, ensure_ascii=False)
    del dev

    rootdir_6 = 'data2/test-perm'  # 2086
    for parent, dirnames, filenames in os.walk(rootdir_6):
        for filename in filenames:
            if not filename[-1] == 't':
                if not filename[0] == '.':
                    filename = os.path.join(parent, filename)
                    firsts, seconds, thirds, fulls, labels = load_data(filename, word2index)
                    test['firsts'].extend(firsts)
                    test['seconds'].extend(seconds)
                    test['thirds'].extend(thirds)
                    test['fulls'].extend(fulls)
                    test['labels'].extend(labels)

    print("Test : size is %d" % len(test['firsts']))

    with open('processed/test.json', 'w') as f:
        json.dump(test, f, ensure_ascii=False)
    del test

    print("See processed folder")
