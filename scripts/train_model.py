# Imports
import argparse
import os
import torch
import yaml
import torch.nn as nn
from models import get_model, unfreeze_model, get_device
from pokenet.dataloader import get_data_loaders

def load_config(path):
    '''
    Load configuration file

    Args:
    path (str): Path to the .yaml config file
    '''
    with open(path, "r", encoding = "utf-8") as file:
        return yaml.safe_load(file)

def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    '''
    Trains the model for a single epoch

    Args:
    model (nn.Module): ResNet50 model
    dataloader (DataLoader): Dataloader for training data
    optimizer (torch.optim.Optimizer): Optimizer (Adam) to update weights
    loss_function (nn.Module): Cross-Entropy Loss to calculate error
    device (torch.device): GPU or CPU, depending on availability
    '''
    # Set the model to training mode
    model.train()
    
    train_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        # Performs a forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)  # Calculates loss
        loss.backward()  # Performs a backward pass
        optimizer.step()  # Update weights from the optimizer
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    average_loss = train_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def evaluate(model, dataloader, loss_function, device):
    '''
    Evaluates the model on validation data w/o updating weights

    Args:
    model (nn.Module): ResNet50 model
    dataloader (DataLoader): Dataloader for evaluating data
    loss_function (nn.Module): Loss function to calculate error
    device (torch.device): GPU or CPU, depending on availability
    '''
    # Set the model to evaluation mode
    model.eval()

    validation_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            validation_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = validation_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def train_model(base_path, config):
    '''
    Goes through Phase 1 (training w/ frozen layers) and Phase 2 (fine-tune the full network

    Args:
    base_path (str): Path to the dataset root directory
    config (dict): Configuration dictionary loaded from config.yaml
    '''
    # Config hyperparameters
    EPOCHS_PHASE1 = config["epochs_phase1"]
    EPOCHS_PHASE2 = config["epochs_phase2"]
    LEARNING_RATE_PHASE1 = config["learning_rate_phase1"]
    LEARNING_RATE_PHASE2 = config["learning_rate_phase2"]
    BATCH_SIZE = config["batch_size"]

    device = get_device()
    model = get_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    train_loader, validation_loader, test_loader = get_data_loaders(base_path, batch_size = BATCH_SIZE)

    # Phase 1: Run and train epoch only on FC layer parameters
    optimizer_1 = torch.optim.Adam(model.fc.parameters(), lr = LEARNING_RATE_PHASE1)
    for epoch in range(EPOCHS_PHASE1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer_1, loss_function, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_function, device)
        print(f"Phase 1 Epoch {epoch + 1} / {EPOCHS_PHASE1}\n Train Loss: {train_loss} | Train Accuracy: {train_accuracy}\n Validation Loss: {validation_loss} | Validation Accuracy: {validation_accuracy}")

    # Phase 2: Fine tune the entire network
    model = unfreeze_model(model)
    optimizer_2 = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE_PHASE2)
    for epoch in range(EPOCHS_PHASE2):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer_2, loss_function, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_function, device)
        print(f"Phase 2 Epoch {epoch + 1} / {EPOCHS_PHASE2}\n Train Loss: {train_loss} | Train Accuracy: {train_accuracy}\n Validation Loss: {validation_loss} | Validation Accuracy: {validation_accuracy}")

    # Save the model
    torch.save(model.state_dict(), "pokenet.pth")
    print("Model saved successfully to pokenet.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train PokéNet Model")
    parser.add_argument("--config", type = str, default = "config.yaml", help = "Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)
    base_path = os.environ["POKEMON_DATASET_PATH"]
    train_model(base_path, config)