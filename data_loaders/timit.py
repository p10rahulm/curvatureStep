# data_loaders/timit.py

import torchaudio
from torchaudio.datasets import TIMIT
from torch.utils.data import DataLoader

def load_timit(batch_size=64):
    train_dataset = TIMIT(root='./data', download=True, subset='train')
    test_dataset = TIMIT(root='./data', download=True, subset='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
