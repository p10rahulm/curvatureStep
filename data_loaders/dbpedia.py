# data_loaders/dbpedia.py

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
    for label, title, content in data_iter:
        yield tokenizer(f"{title} {content}")

# Function to read CSV and yield rows
def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            label = row[0]
            title = row[1]
            content = row[2]
            yield label, title, content

# Load your dataset
train_file_path = 'data/dbpedia/train.csv'
test_file_path = 'data/dbpedia/test.csv'

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
    return int(x) - 1  # Convert labels from 1-14 to 0-13

# Collate function for DataLoader
def collate_batch(batch, vocab=vocab):
    label_list, text_list, lengths = [], [], []
    for label, title, content in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(f"{title} {content}", vocab), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list, label_list, lengths

# Function to load DBpedia dataset
def load_dbpedia(batch_size=1):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader
