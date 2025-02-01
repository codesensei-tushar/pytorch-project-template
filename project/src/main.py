import torch
import yaml
from utils.config import Config
from data.dataset import get_dataloaders
from models.conv import CNN
from train import train_model

def load_config(config_path="project/src/utils/config.yml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def main():
    abc = Config()
    
    # # Initialize model
    # model = CNN(config.num_classes,config).to(config.device)
    config = load_config('project/src/utils/config.yml')
    
    # Instantiate the model and move it to the configured device
    model = CNN(config['model']['num_classes'], config).to(config['config']['device'])
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Train the model
    train_model(model, train_loader, test_loader, config)

if __name__ == "__main__":
    main()