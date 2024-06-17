# Define the relative path to the project root from the current script
import os
import sys
# Add the project root to the system path
project_root =os.getcwd()
sys.path.insert(0, project_root)

from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {epoch_loss/len(train_loader):.4f}")


def train_lm(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0
        for text, labels, lengths in tqdm(train_loader):
            # text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)
            text, labels = text.to(device), labels.to(device)

            def closure():
                optimizer.zero_grad()
                model.flatten_parameters()  # Ensure RNN weights are compacted
                predictions = model(text, lengths).squeeze(1)
                loss = criterion(predictions, labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}')