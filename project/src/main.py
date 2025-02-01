import torch
from utils.config import Config
from data.dataset import get_dataloaders
from models.conv import CNN
from train import train_model

def main():
    config = Config()
    
    # Initialize model
    model = CNN(config.num_classes).to(config.device)
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Train the model
    train_model(model, train_loader, test_loader, config)

if __name__ == "__main__":
    main()