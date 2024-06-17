import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = hidden[-1,:,:]  # Take the last hidden state
        return self.sigmoid(self.fc(hidden))
#
#
# # models/simpleRNN.py
#
# import torch.nn as nn
# import torch
#
# class SimpleRNN(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
#         super().__init__()
#
#         self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
#         self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text, text_lengths):
#         embedded = self.dropout(self.embedding(text))
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
#         packed_output, (hidden, cell) = self.rnn(packed_embedded)
#         hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         return self.fc(hidden)
