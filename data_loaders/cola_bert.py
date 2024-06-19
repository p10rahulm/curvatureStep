# data_loaders/cola_bert.py

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import CoLA


# Define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load your dataset
train_iter, dev_iter, test_iter = CoLA(split=('train', 'dev', 'test'))

# Function to convert text to tensor for BERT
def text_pipeline(x):
    return tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

# Function to convert label to tensor
def label_pipeline(x):
    return int(x)

# Collate function for DataLoader
def collate_batch(batch):
    label_list, input_ids_list, attention_mask_list = [], [], []
    for (_, _label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        input_ids_list.append(processed_text['input_ids'].squeeze(0))
        attention_mask_list.append(processed_text['attention_mask'].squeeze(0))
    
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_list = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'label': label_list
    }

# Function to load CoLA dataset
def load_cola(batch_size=16):
    # Convert iterators to lists for DataLoader
    train_data = list(train_iter)
    # dev_data = list(dev_iter)
    test_data = list(test_iter)

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    # dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader
