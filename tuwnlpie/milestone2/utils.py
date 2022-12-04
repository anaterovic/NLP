import csv
import time
from pathlib import Path

import nltk
import pandas as pd
import torch


# Set the optimizer and the loss function!
# https://pytorch.org/docs/stable/optim.html
import torch.optim as optim
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as split
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from tuwnlpie.milestone2.model import BoWClassifier


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


def read_crowd_truth_csv(
        path_cause=Path('.', 'data', 'crowd_truth_cause.csv'),
        path_treat=Path('.', 'data', 'crowd_truth_treat.csv'),
        shall_sdp=False
    ):

    usedcols = ['sentence', 'term1', 'term2']
    df_cause = pd.read_csv(
        path_cause,
        sep=',', quotechar='"',
        skipinitialspace=True,
        encoding='utf-8',
        on_bad_lines='skip',
        usecols=usedcols
    )
    df_cause["is_cause"] = 1
    df_cause["is_treat"] = 0
    df_treat = pd.read_csv(
        path_treat,
        sep=',', quotechar='"',
        skipinitialspace=True,
        encoding='utf-8',
        on_bad_lines='skip',
        usecols=usedcols
    )
    df_treat["is_treat"] = 1
    df_treat["is_cause"] = 0
    df = df_cause.append(df_treat, ignore_index=True)

    # Make case insensitive (no loss because emphasis on words does not play a role)
    df['sentence'] = df['sentence'].map(lambda x: x.lower())
    # Replace entities in sentence with placeholder tokens (may be useful for generalization when using n-grams)
    df['sentence'] = df.apply(lambda x: x['sentence'].replace(x['term1'].lower(), 'TERM_ONE'), axis=1)
    df['sentence'] = df.apply(lambda x: x['sentence'].replace(x['term2'].lower(), 'TERM_TWO'), axis=1)
    df = df[df['sentence'].apply(lambda x: 'TERM_ONE' in x and 'TERM_TWO' in x)]

    # Convert labels to right dtype
    df['is_cause'] = df['is_cause'].astype(float).astype(int)
    df['is_treat'] = df['is_treat'].astype(float).astype(int)

    # Tokenize the sentences
    df['tokens'] = df['sentence'].apply(lambda x: nltk.word_tokenize(x))
    # Remove stop words and tokens with length smaller than 2 (i.e. punctuations)
    df['tokens'] = df['tokens'].apply(lambda x: [token for token in x if token not in nltk.corpus.stopwords.words('english') and len(token) > 1])
    # Perform stemming
    porter = nltk.PorterStemmer()
    df['tokens_stem'] = df['tokens'].apply(lambda x: [porter.stem(token) for token in x])

    # Perform lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['tokens_lemma'] = df['tokens_stem'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
    
    if shall_sdp:
        # df['sdp_tokens_lemma'] =
        pass
    return df


def split_data_set(df, rate=0.8):
    low_split, heigh_split = split(df, test_size=rate)
    return low_split, heigh_split

def length_longest_sentence(df):
    word_count = lambda sentence: len(nltk.word_tokenize(sentence))
    longest_sentence = max(df, key=word_count)
    length_long_sentence = len(nltk.word_tokenize(longest_sentence))
    return length_long_sentence

class TorchDataset(Dataset):
    def __init__(self, df, feature_cols, label_cols):
        super().__init__()
        self.df = df
        self.feature_cols = feature_cols
        self.label_cols = label_cols
            
    def __getitem__(self, idx):
        x = self.df[self.feature_cols].iloc[idx]
        y = self.df[self.label_cols].iloc[idx]
        return ([torch.tensor(x).float()], torch.tensor(y).long())
    
    def __len__(self):
        return len(self.df)

    def get_dataloader(self, batch_size=128, num_workers=8, shuffle=False):
        return DataLoader(self, batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=num_workers, shuffle=shuffle)










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

class TorchTrainer():
    def __init__(self, model, name, dirpath, dataloaders, max_epochs=50) -> None:
        self.model = model
        self.name = name
        self.dirpath = dirpath
        self.max_epochs = max_epochs
        self.dataloaders = dataloaders

    def train(self):
        logger = TensorBoardLogger(f"{self.dirpath}/tensorboard", name=self.name)
        callbacks = [
            ModelCheckpoint(dirpath=Path(self.dirpath, self.name), monitor="val_loss"),
            EarlyStopping(monitor='loss')]
        trainer = Trainer(deterministic=True, logger=logger, callbacks=callbacks, max_epochs=self.max_epochs)
        trainer.fit(self.model, self.dataloaders['train'], self.dataloaders['val'])
        val_result = trainer.test(self.model, self.dataloaders['val'], verbose=True)
        test_result = trainer.test(self.model, self.dataloaders['test'], verbose=True)
        result = {"test_acc": test_result, "val_acc": val_result}
        #return result
