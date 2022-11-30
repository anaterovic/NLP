from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def read_food_disease_csv(path=Path('..', 'data', 'food_disease.csv')):
    usecols = ['food_entity', 'disease_entity', 'sentence', 'is_cause', 'is_treat']
    df = pd.read_csv(
        path, sep=',', quotechar='"', skipinitialspace=True,
        encoding='utf-8', on_bad_lines='skip', usecols=usecols
    )
    df['sentence'] = df['sentence'].map(lambda x: x.lower())
    # Replace entities in sentence with placeholder tokens (may be useful for generalization when using n-grams)
    df['sentence'] = df.apply(lambda x: x['sentence'].replace(x['food_entity'], 'food_entity'), axis=1)
    df['sentence'] = df.apply(lambda x: x['sentence'].replace(x['disease_entity'], 'disease_entity'), axis=1)
    # Drop malformed documents (both entities must be present in sentence)
    df = df[df['sentence'].apply(lambda x: 'food_entity' in x and 'disease_entity' in x)]
    # Convert labels to right dtype
    df['is_cause'] = df['is_cause'].astype(float).astype(int)
    df['is_treat'] = df['is_treat'].astype(float).astype(int)
    return df


def split_data(docs):
    train, test = train_test_split(docs, test_size=0.20)
    return train, test


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