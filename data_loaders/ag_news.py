# data_loaders/ag_news.py

from torchtext.datasets import AG_NEWS
from torchtext.data import Field, LabelField, BucketIterator
import torch

def load_ag_news(batch_size=64, device='cpu'):
    TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
    LABEL = LabelField(dtype=torch.float)

    train_data, test_data = AG_NEWS.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data, max_size=25000)
    LABEL.build_vocab(train_data)

    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        device=device
    )

    return train_iterator, test_iterator, TEXT, LABEL
