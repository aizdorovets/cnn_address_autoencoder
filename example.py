import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.cnn_autoencoder import CNNAutoencoder
from src.bpe_tokenizer import BPETokenizer
from utils.data_utils import build_data_loader
from utils.utils import get_device, infer


# Config and device
config_path = 'models/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
device = get_device()
print(f'Using {device}')

# Instantiate the model
model = CNNAutoencoder(config, device)
model.to(device)
# Load weights
model_path = 'models/address_ae.pth'
model.load_state_dict(torch.load(model_path))
# Tokenizer
tokenizer = BPETokenizer('models/augmented_bpe_5000.model', config['max_len'])

# Training stuff
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

# Build data loader
with open('data/augmented_address_dataset.pickle', 'rb') as f:
    data = pickle.load(f)
train_idx = int(np.ceil(len(data) * 0.8))
train_data = data[:train_idx]
val_data = data[train_idx:]
y_train = [0] * len(train_data)
y_val = [0] * len(val_data)

# Train
# Epochs
n_epochs = config['num_epochs']

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0
    test_loss = 0.0

    train_loader = build_data_loader(
        train_data,
        y_train,
        config['train_batch_size'],
        tokenizer,
    )
    test_loader = build_data_loader(
        val_data,
        y_val,
        config['test_batch_size'],
        tokenizer,
    )

    # # Training
    # model.train()
    # for batch in tqdm(train_loader):
    #     address = batch
    #
    #     address = address.to(device)
    #     optimizer.zero_grad()
    #     logits, probs = model(address)
    #     loss = loss_function(logits, address)
    #     loss.backward()
    #     optimizer.step()
    #     train_loss += loss.item()
    #
    # train_loss = train_loss / len(train_loader)
    # print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # Test
    model.eval()
    probabilities = []
    candidates = []
    for batch in tqdm(test_loader):
        address = batch
        address = address.to(device)
        with torch.no_grad():
            logits, probs = model(address)
            loss = loss_function(logits, address)
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    print('Epoch: {} \tTest Loss: {:.6f}'.format(epoch, test_loss))
    print('\n')

# Assuming there is a model and a tokenizer loaded
model.eval()
inputs = [
    'Москва, улица Пушкина, дом 69',
    'Алтайский край, р-н. Михайловский, с. Михайловское, ул. Западная, 57',
    'ул. Алексеева, 50, Красноярск, Красноярский край, 660077',
    'просп. Металлургов 25, Темиртау 100000'
]
preds = infer(inputs, model, tokenizer, device)
for inp, pred in zip(inputs, preds):
    print(f'Input: {inp}')
    print(f'Pred: {pred}')
    print('\n')
