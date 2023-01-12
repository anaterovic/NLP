from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, matmul
from torch.nn.functional import softmax
from torchmetrics import Accuracy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_position_embedding import PositionEmbedding


from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

device = torch.device("cpu")
from torch_position_embedding import PositionEmbedding

class TorchModel(LightningModule):
    def __init__(self, learning_rate=1e-2) -> None:
        super().__init__()
        self.save_hyperparameters('learning_rate')

        self.wordEmbeddings = nn.Embedding(11212,110)
        self.positionEmbeddings = nn.Embedding(110,40)
        # self.positionEmbeddings = PositionEmbedding(num_embeddings=11212, embedding_dim=110, mode=PositionEmbedding.MODE_ADD)
        self.transformerLayer = nn.TransformerEncoderLayer(150,15) #this transofrmer contains muti head attention
        self.linear1 = nn.Linear(150, 64)
        self.linear2 = nn.Linear(64, 1)
        self.linear3 = nn.Linear(110,  16)
        self.linear4 = nn.Linear(16, 2)
           
    def forward(self, x):
        positions = (torch.arange(0,110).reshape(1,110) + torch.zeros(x.shape[0],110)).to(device)
        sentence = torch.cat((self.wordEmbeddings(x.long()).squeeze(2),self.positionEmbeddings(positions.long())),axis=2)
        attended = self.transformerLayer(sentence)
        linear1 = F.relu(self.linear1(attended))
        linear2 = torch.sigmoid(self.linear2(linear1))
        linear2 = linear2.view(-1,110) # reshaping the layer as the transformer outputs a 2d tensor (or 3d considering the batch size)
        linear3 = F.relu(self.linear3(linear2))
        out = torch.sigmoid(self.linear4(linear3))
        return out
    
    def _loss_fn(self, out, y):
        loss = F.binary_cross_entropy(out, y) # Multiclass classification
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.squeeze()
        loss = self._loss_fn(out, y.float())
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)      
        return loss

    def test_step(self, batch, batch_idx):
        print("TEST DATA")
        with torch.no_grad():
            x, y = batch
            out = self(x)
            out = out.squeeze()
            loss = self._loss_fn(out, y.float())
            report = classification_report(np.argmax(y, axis=1),np.argmax(out, axis=1),target_names=['is_cause', 'is_treat'])
            print(report)
            
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            out = self(x)
            out = out.squeeze()
            loss = self._loss_fn(out, y.float())
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return torch.optim.Adagrad(
            self.parameters(), lr=self.hparams.learning_rate)
        


# class TorchModel(LightningModule):
#     def __init__(self, learning_rate=1e-2) -> None:
#         super().__init__()
#         self.save_hyperparameters('learning_rate')

#         self.wordEmbeddings = nn.Embedding(11212, 110)
#         self.positionEmbeddings = nn.Embedding(110, 40)
#         # self.positionEmbeddings = PositionEmbedding(num_embeddings=11212, embedding_dim=110, mode=PositionEmbedding.MODE_ADD)
#         self.transformerLayer = nn.TransformerEncoderLayer(150, 15)
#         self.linear1 = nn.Linear(150, 64)
#         self.linear2 = nn.Linear(64, 1)
#         self.linear3 = nn.Linear(110, 16)
#         self.linear4 = nn.Linear(16, 2)

#     def forward(self, x):
#         positions = (torch.arange(0, 110).reshape(1, 110) + torch.zeros(x.shape[0], 110)).to(device)
#         sentence = torch.cat((self.wordEmbeddings(x.long()).squeeze(2), self.positionEmbeddings(positions.long())),
#                              axis=2)
#         attended = self.transformerLayer(sentence)
#         linear1 = F.relu(self.linear1(attended))
#         linear2 = torch.sigmoid(self.linear2(linear1))
#         linear2 = linear2.view(-1,
#                                110)  # reshaping the layer as the transformer outputs a 2d tensor (or 3d considering the batch size)
#         linear3 = F.relu(self.linear3(linear2))
#         out = torch.sigmoid(self.linear4(linear3))
#         return out

#     def _loss_fn(self, out, y):
#         loss = F.binary_cross_entropy(out, y)  # Multiclass classification
#         return loss

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         out = self(x)
#         out = out.squeeze()
#         loss = self._loss_fn(out, y.float())
#         self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         print("test step")
#         with torch.no_grad():
#             x, y = batch
#             out = self(x)
#             out = out.squeeze()
#             loss = self._loss_fn(out, y.float())
#             self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#             accuracy = Accuracy()
#             acc = accuracy(out, y)
#             self.log('accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

#     def validation_step(self, batch, batch_idx):
#         print("validation step")
#         with torch.no_grad():
#             x, y = batch
#             out = self(x)
#             out = out.squeeze()
#             loss = self._loss_fn(out, y.float())
#             self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#             accuracy = Accuracy()
#             acc = accuracy(out, y)
#             self.log('accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

#     def configure_optimizers(self):
#         return torch.optim.Adagrad(
#             self.parameters(), lr=self.hparams.learning_rate)

