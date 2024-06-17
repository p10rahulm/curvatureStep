import torch.optim as optim
import tqdm

from models.simpleRNN import SimpleRNN
from data_loaders.imdb import vocab
import torch.nn as nn
from data_loaders.imdb import load_imdb_reviews

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f"epoch number = {epoch+1}")
        epoch_loss = 0
        for text, labels, lengths in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')

# Hyperparameters
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 256
output_dim = 1
pad_idx = vocab["<pad>"]

# Initialize model, criterion, optimizer
model = SimpleRNN(vocab_size, embed_dim, hidden_dim, output_dim, pad_idx)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Load data
train_loader, test_loader = load_imdb_reviews()

print("Started Training")
# Train the model
train_model(model, train_loader, criterion, optimizer)
