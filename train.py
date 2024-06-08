import torch
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {running_loss/len(train_loader):.4f}")
