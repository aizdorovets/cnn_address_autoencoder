import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim


class CNNAutoencoder(nn.Module):

    def __init__(self, params, device):
        super(CNNAutoencoder, self).__init__()
        self.params = params
        self.device = device
        self.botneck1 = 128
        self.botneck2 = 64

        # utils
        self.fc = nn.Linear(params['dim'], params['vocab_size'])
        self.dropout = nn.Dropout(p=0.1)

        # Embedding
        self.embedding = nn.Embedding(params['vocab_size'], params['dim'])

        # Encoder
        self.conv1 = nn.Conv1d(params['dim'], self.botneck1, 5, padding=1)
        self.conv2 = nn.Conv1d(self.botneck1, self.botneck2, 5, padding=1)
        # self.pool = nn.MaxPool1d(2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose1d(
            self.botneck2,
            self.botneck1,
            5,
            stride=1,
            padding=1,
        )
        self.t_conv2 = nn.ConvTranspose1d(
            self.botneck1,
            params['dim'],
            5,
            stride=1,
            padding=1,
        )

    def encode(self, x):
        x = self.embedding(x)  # BS Len Dim
        x = x.permute(0, 2, 1)  # BS Dim Len
        # Conv
        # 1st
        x = self.conv1(x)
        x = nn.BatchNorm1d(self.botneck1, device=self.device)(x)
        x = F.relu(x)
        x = self.dropout(x)
        # 2nd
        x = self.conv2(x)
        x = nn.BatchNorm1d(self.botneck2, device=self.device)(x)
        x = F.relu(x)
        # x = self.dropout(x)
        return x

    def decode(self, x):
        # Deconv
        # 1st
        x = self.t_conv1(x)
        x = nn.BatchNorm1d(self.botneck1, device=self.device)(x)
        x = F.relu(x)

        # 2nd
        x = self.t_conv2(x)
        x = nn.BatchNorm1d(self.params['dim'], device=self.device)(x)
        x = F.relu(x)
        return x

    def lm_head(self, x):
        # FC
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        logits = F.sigmoid(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        logits, probs = self.lm_head(x)
        return logits, probs
