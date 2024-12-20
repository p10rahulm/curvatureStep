# Define the relative path to the project root from the current script
import os
import sys
# Add the project root to the system path
project_root =os.getcwd()
sys.path.insert(0, project_root)

import torch
from tqdm import tqdm

def test_with_logging_base(model, test_loader, criterion, device):
    """Base testing function for standard neural networks with logging"""
    model.eval()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)
    batch_count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            batch_count += 1

    avg_test_loss = test_loss / batch_count
    accuracy = correct / total
    
    print(f"Test set: Average loss: {avg_test_loss:.4f}, "
          f"Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)")
    
    return avg_test_loss, accuracy

def test_with_logging_lm(model, test_loader, criterion, device):
    """Testing function for binary classification language models with logging"""
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    batch_count = 0
    
    with torch.no_grad():
        for text, labels, lengths in tqdm(test_loader):
            text, labels = text.to(device), labels.to(device)
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += len(labels)
            batch_count += 1
    
    avg_test_loss = test_loss / batch_count
    accuracy = correct_preds / total_preds
    
    print(f"Test set: Average loss: {avg_test_loss:.4f}, "
          f"Accuracy: {correct_preds}/{total_preds} ({100. * accuracy:.2f}%)")
    
    return avg_test_loss, accuracy

def test_with_logging_lm_multiclass(model, test_loader, criterion, device):
    """Testing function for multiclass language models with logging"""
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    batch_count = 0
    
    with torch.no_grad():
        for text, labels, lengths in tqdm(test_loader):
            text, labels = text.to(device), labels.to(device)
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += len(labels)
            batch_count += 1
    
    avg_test_loss = test_loss / batch_count
    accuracy = correct_preds / total_preds
    
    print(f"Test set: Average loss: {avg_test_loss:.4f}, "
          f"Accuracy: {correct_preds}/{total_preds} ({100. * accuracy:.2f}%)")
    
    return avg_test_loss, accuracy

def test_with_logging_bert(model, test_loader, criterion, device):
    """Testing function for BERT models with logging"""
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += len(labels)
            batch_count += 1
    
    avg_test_loss = test_loss / batch_count
    accuracy = correct_preds / total_preds
    
    print(f"Test set: Average loss: {avg_test_loss:.4f}, "
          f"Accuracy: {correct_preds}/{total_preds} ({100. * accuracy:.2f}%)")
    
    return avg_test_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return 1.0 * correct / len(test_loader.dataset)

def test_lm(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    epoch_loss =0.0
    correct_preds=0
    total_preds = 0
    with torch.no_grad():
        for text, labels, lengths in tqdm(test_loader):
            text, labels = text.to(device), labels.to(device)
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += len(labels)
    epoch_loss /= len(test_loader.dataset)
    accuracy =  100. * correct_preds / total_preds
    print(f"Test set: Average loss: {epoch_loss:.4f}, Accuracy: {correct_preds}/{total_preds} ({accuracy:.2f}%)")
    return 1.0 * correct_preds / total_preds

def test_lm_multiclass(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    epoch_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for text, labels, lengths in tqdm(test_loader):
            text, labels = text.to(device), labels.to(device)
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += len(labels)
    epoch_loss /= len(test_loader.dataset)
    accuracy = 100. * correct_preds / total_preds
    print(f"Test set: Average loss: {epoch_loss:.4f}, Accuracy: {correct_preds}/{total_preds} ({accuracy:.2f}%)")
    return 1.0 * correct_preds / total_preds


def test_bert(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    epoch_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += len(labels)
    epoch_loss /= len(test_loader.dataset)
    accuracy = 100. * correct_preds / total_preds
    print(f"Test set: Average loss: {epoch_loss:.4f}, Accuracy: {correct_preds}/{total_preds} ({accuracy:.2f}%)")
    return 1.0 * correct_preds / total_preds

