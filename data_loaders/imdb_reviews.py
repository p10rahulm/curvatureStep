import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Define a function to yield tokens from the dataset
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# Load your dataset
train_iter, test_iter = IMDB(split=('train', 'test'))

# Build the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])


# Function to convert text to tensor
def text_pipeline(x):
    return vocab(tokenizer(x))


# Function to convert label to tensor
def label_pipeline(x):
    return 1 if x == 'pos' else 0


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list, label_list, lengths

def load_imdb_reviews(batch_size=64):
    # Define tokenizer
    # Convert iterators to lists for DataLoader
    train_data = list(train_iter)
    test_data = list(test_iter)

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader

#
#
# # data_loaders/imdb_reviews.py
#
# from torchtext.datasets import IMDB
# from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
# import torch
#
# def load_imdb_reviews(batch_size=64, device='cpu'):
#     TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
#     LABEL = LabelField(dtype=torch.float)
#
#     train_data, test_data = IMDB.splits(TEXT, LABEL)
#
#     TEXT.build_vocab(train_data, max_size=25000)
#     LABEL.build_vocab(train_data)
#
#     train_iterator, test_iterator = BucketIterator.splits(
#         (train_data, test_data),
#         batch_size=batch_size,
#         sort_within_batch=True,
#         device=device
#     )
#
#     return train_iterator, test_iterator, TEXT, LABEL
