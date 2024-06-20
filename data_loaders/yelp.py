import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
import csv

# Load GloVe embeddings
glove = GloVe(name='6B', dim=100)

# Define tokenizer
tokenizer = get_tokenizer('basic_english')

# Determine vocabulary size from GloVe
vocab_size = len(glove.stoi)
pad_idx = glove.stoi['pad']

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
    return iter(read_csv(file_path))

# Function to convert text to tensor using GloVe vocabulary
def text_pipeline(x):
    return [glove.stoi[token] if token in glove.stoi else glove.stoi['unk'] for token in tokenizer(x)]

# Function to convert label to tensor
def label_pipeline(x):
    return int(x) - 1  # Convert labels to zero-index

# Collate function for DataLoader
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for label, content in batch:
        label_list.append(label_pipeline(label))
        tokens = text_pipeline(content)
        processed_text = torch.tensor(tokens, dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=glove.stoi['pad'])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    
    return text_list, label_list, lengths

# Function to create DataLoader from CSV
def create_dataloader(file_path, batch_size=64):
    return DataLoader(list(read_csv(file_path)), batch_size=batch_size, collate_fn=collate_batch)

# Function to load Yelp dataset
def load_yelp(batch_size=1024):
    train_loader = create_dataloader(train_file_path, batch_size=batch_size)
    test_loader = create_dataloader(test_file_path, batch_size=batch_size)
    
    return train_loader, test_loader
