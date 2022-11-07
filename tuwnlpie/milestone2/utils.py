import csv
import time

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


class Trainer:
    def __init__(
        self,
        dataset: IMDBDataset,
        model: BoWClassifier,
        model_path: str = None,
        test: bool = False,
        lr: float = 0.001,
    ):
        self.dataset = dataset
        self.model = model

        # The optimizer will update the weights of our model based on the loss function
        # This is essential for correct training
        # The _lr_ parameter is the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()

        # Copy the model and the loss function to the correct device
        self.model = self.model.to(dataset.device)
        self.criterion = self.criterion.to(dataset.device)

    def calculate_performance(self, preds, y):
        """
        Returns precision, recall, fscore per batch
        """
        # Get the predicted label from the probabilities
        rounded_preds = preds.argmax(1)

        # Calculate the correct predictions batch-wise and calculate precision, recall, and fscore
        # WARNING: Tensors here could be on the GPU, so make sure to copy everything to CPU
        precision, recall, fscore, support = precision_recall_fscore_support(
            rounded_preds.cpu(), y.cpu()
        )

        return precision[1], recall[1], fscore[1]

    def train(self, iterator):
        # We will calculate loss and accuracy epoch-wise based on average batch accuracy
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0

        # You always need to set your model to training mode
        # If you don't set your model to training mode the error won't propagate back to the weights
        self.model.train()

        # We calculate the error on batches so the iterator will return matrices with shape [BATCH_SIZE, VOCAB_SIZE]
        for batch in iterator:
            text_vecs = batch[0]
            labels = batch[1]
            sen_lens = []
            texts = []

            # This is for later!
            if len(batch) > 2:
                sen_lens = batch[2]
                texts = batch[3]

            # We reset the gradients from the last step, so the loss will be calculated correctly (and not added together)
            self.optimizer.zero_grad()

            # This runs the forward function on your model (you don't need to call it directly)
            predictions = self.model(text_vecs, sen_lens)

            # Calculate the loss and the accuracy on the predictions (the predictions are log probabilities, remember!)
            loss = self.criterion(predictions, labels)

            prec, recall, fscore = self.calculate_performance(predictions, labels)

            # Propagate the error back on the model (this means changing the initial weights in your model)
            # Calculate gradients on parameters that requries grad
            loss.backward()
            # Update the parameters
            self.optimizer.step()

            # We add batch-wise loss to the epoch-wise loss
            epoch_loss += loss.item()
            # We also do the same with the scores
            epoch_prec += prec.item()
            epoch_recall += recall.item()
            epoch_fscore += fscore.item()
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    # The evaluation is done on the validation dataset
    def evaluate(self, iterator):

        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0
        # On the validation dataset we don't want training so we need to set the model on evaluation mode
        self.model.eval()

        # Also tell Pytorch to not propagate any error backwards in the model or calculate gradients
        # This is needed when you only want to make predictions and use your model in inference mode!
        with torch.no_grad():

            # The remaining part is the same with the difference of not using the optimizer to backpropagation
            for batch in iterator:
                text_vecs = batch[0]
                labels = batch[1]
                sen_lens = []
                texts = []

                if len(batch) > 2:
                    sen_lens = batch[2]
                    texts = batch[3]

                predictions = self.model(text_vecs, sen_lens)
                loss = self.criterion(predictions, labels)

                prec, recall, fscore = self.calculate_performance(predictions, labels)

                epoch_loss += loss.item()
                epoch_prec += prec.item()
                epoch_recall += recall.item()
                epoch_fscore += fscore.item()

        # Return averaged loss on the whole epoch!
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    def training_loop(self, train_iterator, valid_iterator, epoch_number=15):
        # Set an EPOCH number!
        N_EPOCHS = epoch_number

        best_valid_loss = float("inf")

        # We loop forward on the epoch number
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            # Train the model on the training set using the dataloader
            train_loss, train_prec, train_rec, train_fscore = self.train(train_iterator)
            # And validate your model on the validation set
            valid_loss, valid_prec, valid_rec, valid_fscore = self.evaluate(
                valid_iterator
            )

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # If we find a better model, we save the weights so later we may want to reload it
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {train_loss:.3f} | Train Prec: {train_prec*100:.2f}% | Train Rec: {train_rec*100:.2f}% | Train Fscore: {train_fscore*100:.2f}%"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val Prec: {valid_prec*100:.2f}% | Val Rec: {valid_rec*100:.2f}% | Val Fscore: {valid_fscore*100:.2f}%"
            )

        return best_valid_loss
