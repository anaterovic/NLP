import csv

import nltk
import numpy as np
from tqdm import tqdm


def read_docs_from_csv(filename):
    docs = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for text, label in tqdm(reader):
            words = nltk.word_tokenize(text)
            docs.append((words, label))

    return docs


def split_train_dev_test(docs, train_ratio=0.8, dev_ratio=0.1):
    np.random.seed(2022)
    np.random.shuffle(docs)
    train_size = int(len(docs) * train_ratio)
    dev_size = int(len(docs) * dev_ratio)
    return (
        docs[:train_size],
        docs[train_size : train_size + dev_size],
        docs[train_size + dev_size :],
    )


def calculate_tp_fp_fn(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp += 1
        else:
            if true == "positive":
                fn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, fscore
