import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.datasets import reuters

# Load the Reuters dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Function to collate batches
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for text, label in batch:
        label_list.append(label)
        processed_text = torch.tensor(text, dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list, label_list, lengths

# Function to load Reuters dataset
def load_reuters(batch_size=64):
    # Create DataLoader
    train_data_combined = list(zip(train_data, train_labels))
    test_data_combined = list(zip(test_data, test_labels))

    train_loader = DataLoader(train_data_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data_combined, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader

