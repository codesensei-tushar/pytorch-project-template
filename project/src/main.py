import torch
import yaml
import importlib
from utils.config import Config
from models.conv import CNN
from train import train_model

def load_config(config_path="project/src/utils/config.yml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# from data.dataset import get_dataloaders

def main():
    abc = Config() 
    # # Initialize model
    # model = CNN(config.num_classes,config).to(config.device)
    config = load_config('project/src/utils/config.yml')
    
    # Instantiate the model and move it to the configured device
    model = CNN(config['model']['num_classes'], config).to(config['config']['device'])
    
    # Get data loaders
    module_name = config["dataloader"]["module"]
    function_name = config["dataloader"]["function"]
    # Dynamically import the module
    dataset_module = importlib.import_module(module_name)
    get_dataloaders = getattr(dataset_module, function_name)

    train_loader, test_loader = get_dataloaders(config)
    
    # Train the model
    train_model(model, train_loader, test_loader, config)

if __name__ == "__main__":
    main()