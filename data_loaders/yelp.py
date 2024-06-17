# data_loaders/yelp.py

import os
import csv
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Define a function to yield tokens from the dataset
def yield_tokens(data_iter):
    for label, content in data_iter:
        yield tokenizer(content)

# Function to read CSV and yield rows
def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            label = row[0]
            content = row[1]
            yield label, content

# Load your dataset
train_file_path = 'data/yelp/train.csv'
test_file_path = 'data/yelp/test.csv'

def get_data_iter(file_path):
    data = list(read_csv(file_path))
    return iter(data), data

train_iter, train_data = get_data_iter(train_file_path)
test_iter, test_data = get_data_iter(test_file_path)

# Build the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Function to convert text to tensor
def text_pipeline(x, vocab):
    return vocab(tokenizer(x))

# Function to convert label to tensor
def label_pipeline(x):
    return int(x) - 1  # Convert labels to zero-index

# Collate function for DataLoader
def collate_batch(batch, vocab=vocab):
    label_list, text_list, lengths = [], [], []
    for label, content in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(content, vocab), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list, label_list, lengths

# Function to load Yelp dataset
def load_yelp(batch_size=64):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader
