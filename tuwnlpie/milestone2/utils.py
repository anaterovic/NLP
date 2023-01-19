import csv
import time
import numpy as np
from pathlib import Path

import nltk
from nltk import RegexpTokenizer

import spacy
import networkx as nx

import pandas as pd
import torch


# Set the optimizer and the loss function!
# https://pytorch.org/docs/stable/optim.html
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as split
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from keras.preprocessing.text import one_hot
import copy
from keras_preprocessing.sequence import pad_sequences


# This is just for measuring training time!
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def shortest_dep_path(nlp, sentence):
    doc = nlp(sentence)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((
                '{0}'.format(token.lemma_),
                '{0}'.format(child.lemma_)))
    graph = nx.Graph(edges)
    entity1 = 'TERMONE'
    entity2 = 'TERMTWO'
    try:
        return nx.shortest_path(graph, source=entity1, target=entity2)
    except:
        return []

def remove_stop_words(tokens):
    return [x for x in tokens if x not in nltk.corpus.stopwords.words('english') and len(x) > 1]


def read_and_prepare_data(path, shall_sdp=False):

    usedcols = ['sentence', 'term1', 'term2', 'is_cause', 'is_treat']
    df = pd.read_csv(
        path,
        sep=',', quotechar='"',
        skipinitialspace=True,
        encoding='utf-8',
        on_bad_lines='skip',
        usecols=usedcols)

    print("\tData read in - preprocessing started")
    # Make case insensitive (no loss because emphasis on words does not play a role)
    df['sentence'] = df['sentence'].map(lambda x: x.lower())
    # Replace entities in sentence with placeholder tokens (may be useful for generalization when using n-grams)
    df['sentence'] = df.apply(lambda x: x['sentence'].replace(x['term1'].lower(), 'TERMONE').replace('TERMONEs', 'TERMONE'), axis=1)
    df['sentence'] = df.apply(lambda x: x['sentence'].replace(x['term2'].lower(), 'TERMTWO').replace('TERMTWOs', 'TERMTWO'), axis=1)

    df = df[df['sentence'].apply(lambda x: 'TERMONE' in x and 'TERMTWO' in x)]

    # Convert labels to right dtype
    df['is_cause'] = df['is_cause'].astype(float).astype(int)
    df['is_treat'] = df['is_treat'].astype(float).astype(int)

    # Tokenize the sentences
    tokenizer = RegexpTokenizer(r'\w+')
    df['tokens'] = df['sentence'].apply(lambda x: tokenizer.tokenize(x))
    # Remove stop words and tokens with length smaller than 2 (i.e. punctuations)
    df['tokens'] = df['tokens'].apply(lambda x: [token for token in x if token not in nltk.corpus.stopwords.words('english') and len(token) > 1])
    # Perform stemming
    porter = nltk.PorterStemmer()
    df['tokens_stem'] = df['tokens'].apply(lambda x: [porter.stem(token) for token in x])
    
    # Perform lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['tokens_lemma'] = df['tokens_stem'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

    nlp = spacy.load("en_core_web_sm")

    if shall_sdp:
        print("\tstarting shortest path search and stopword removal")
        df['sdp_tokens_lemma'] = df['sentence'].apply(lambda x: remove_stop_words(shortest_dep_path(nlp, x)))
    print("## Finished reading and preparing the data ##")
    return df


def split_data_set(X, y, test_size=0.8, random_state=412):
    s1, s2, s3, s4 = split(X, y, test_size=test_size, random_state=random_state)
    return s1, s2, s3, s4

def length_longest_sentence(df):
    word_count = lambda sentence: len(nltk.word_tokenize(sentence))
    longest_sentence = max(df, key=word_count)
    length_long_sentence = len(nltk.word_tokenize(longest_sentence))
    return length_long_sentence

def encodeX(df):
    unique_words = set()
    longest_sentence = 0
    for sentence in df['tokens']:
        current_sentence = 0
        for word in sentence:
            current_sentence += 1
            if word not in unique_words:
                unique_words.add(word)
            if current_sentence > longest_sentence:
                longest_sentence = current_sentence

    X_tmp = []
    for sentence in df['tokens_lemma']:
        sen_tmp = []
        for token in sentence:
            sen_tmp.append(one_hot(token, len(unique_words)))
        X_tmp.append(sen_tmp)
    
    X_tmp = pad_sequences(X_tmp, longest_sentence, padding='post') 
    # makes all sentences the same length by padding with preset value at the end

    return X_tmp


class TorchDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
            
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (torch.tensor(x).float(), torch.tensor(y).long())
    
    def __len__(self):
        return len(self.x)

    def get_dataloader(self, batch_size=128, num_workers=0, shuffle=False):
        return DataLoader(self, batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=num_workers, shuffle=shuffle)



class TorchTrainer():
    def __init__(self, model, name, dirpath, dataloaders, max_epochs=50) -> None:
        self.model = model
        self.name = name
        self.dirpath = dirpath
        self.max_epochs = max_epochs
        self.dataloaders = dataloaders

    def run(self):
        logger = TensorBoardLogger(f"{self.dirpath}/tensorboard", name=self.name)
        callbacks = [
            ModelCheckpoint(dirpath=Path(self.dirpath, self.name), monitor="val_loss"),
            EarlyStopping(monitor='loss')
            ]
        trainer = Trainer(deterministic=True, logger=logger, callbacks=callbacks, max_epochs=self.max_epochs)
        trainer.fit(self.model, self.dataloaders['train'], self.dataloaders['val'])
        return trainer






# class TorchTrainer():
#     def __init__(self, model, name, dirpath, dataloaders, max_epochs=50) -> None:
#         self.model = model
#         self.name = name
#         self.dirpath = dirpath
#         self.max_epochs = max_epochs
#         self.dataloaders = dataloaders

#     def train(self):
#         logger = TensorBoardLogger(f"{self.dirpath}/tensorboard", name=self.name)
#         callbacks = [
#             ModelCheckpoint(dirpath=Path(self.dirpath, self.name), monitor="val_loss"),
#             EarlyStopping(monitor='loss')
#             ]
#         trainer = Trainer(deterministic=True, logger=logger,
#                           callbacks=callbacks, max_epochs=self.max_epochs)
#         trainer.fit(self.model, self.dataloaders['train'])



class IMDBDataset:
    def __init__(self, data_path, BATCH_SIZE=64):
        self.data_path = data_path
        # Initialize the correct device
        # It is important that every array should be on the same device or the training won't work
        # A device could be either the cpu or the gpu if it is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = BATCH_SIZE

        self.df = self.read_df_from_csv(self.data_path)

        self.transform(self.df)

        self.tr_data, self.val_data, self.te_data = self.split_data(self.df)

        self.word_to_ix = self.prepare_vectorizer(self.tr_data)
        self.VOCAB_SIZE = len(self.word_to_ix.vocabulary_)
        self.OUT_DIM = 2

        (
            self.tr_data_loader,
            self.val_data_loader,
            self.te_data_loader,
        ) = self.prepare_dataloader(
            self.tr_data, self.val_data, self.te_data, self.word_to_ix, self.device
        )

        (
            self.train_iterator,
            self.valid_iterator,
            self.test_iterator,
        ) = self.create_dataloader_iterators(
            self.tr_data_loader,
            self.val_data_loader,
            self.te_data_loader,
            self.BATCH_SIZE,
        )

    def read_df_from_csv(self, filename):
        docs = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for text, label in tqdm(reader):
                docs.append((text, label))

        df = pd.DataFrame(docs, columns=["text", "label"])

        return df

    def transform(self, df):
        labels = {"negative": 0, "positive": 1}

        df["label"] = [labels[item] for item in df.label]

        return df

    def split_data(self, train_data, random_seed=2022):
        tr_data, val_data = split(train_data, test_size=0.2, random_state=random_seed)
        tr_data, te_data = split(tr_data, test_size=0.2, random_state=random_seed)

        return tr_data, val_data, te_data

    def prepare_vectorizer(self, tr_data):
        vectorizer = CountVectorizer(
            max_features=3000, tokenizer=LemmaTokenizer(), stop_words="english"
        )

        word_to_ix = vectorizer.fit(tr_data.text)

        return word_to_ix

    # Preparing the data loaders for the training and the validation sets
    # PyTorch operates on it's own datatype which is very similar to numpy's arrays
    # They are called Torch Tensors: https://pytorch.org/docs/stable/tensors.html
    # They are optimized for training neural networks
    def prepare_dataloader(self, tr_data, val_data, te_data, word_to_ix, device):
        # First we transform the text into one-hot encoded vectors
        # Then we create Torch Tensors from the list of the vectors
        # It is also inportant to send the Tensors to the correct device
        # All of the tensors should be on the same device when training
        tr_data_vecs = torch.FloatTensor(
            word_to_ix.transform(tr_data.text).toarray()
        ).to(device)
        tr_labels = torch.LongTensor(tr_data.label.tolist()).to(device)

        val_data_vecs = torch.FloatTensor(
            word_to_ix.transform(val_data.text).toarray()
        ).to(device)
        val_labels = torch.LongTensor(val_data.label.tolist()).to(device)

        te_data_vecs = torch.FloatTensor(
            word_to_ix.transform(te_data.text).toarray()
        ).to(device=device)
        te_labels = torch.LongTensor(te_data.label.tolist()).to(device=device)

        tr_data_loader = [
            (sample, label) for sample, label in zip(tr_data_vecs, tr_labels)
        ]
        val_data_loader = [
            (sample, label) for sample, label in zip(val_data_vecs, val_labels)
        ]

        te_data_loader = [
            (sample, label) for sample, label in zip(te_data_vecs, te_labels)
        ]

        return tr_data_loader, val_data_loader, te_data_loader

    # The DataLoader(https://pytorch.org/docs/stable/data.html) class helps us to prepare the training batches
    # It has a lot of useful parameters, one of it is _shuffle_ which will randomize the training dataset in each epoch
    # This can also improve the performance of our model
    def create_dataloader_iterators(
        self, tr_data_loader, val_data_loader, te_data_loader, BATCH_SIZE
    ):
        train_iterator = DataLoader(
            tr_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        valid_iterator = DataLoader(
            val_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        test_iterator = DataLoader(
            te_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        return train_iterator, valid_iterator, test_iterator
