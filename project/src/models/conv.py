import torch.nn as nn
import yaml
import os

# Load YAML config from the correct path
config_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'config.yml')
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# class CNN(nn.Module):
#     def __init__(self, num_classes):
#         super(CNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(32, 64, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(64, 128, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 17 * 17, 512),  # Calculated based on input size 150x150
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

class CNN(nn.Module):
    def __init__(self, num_classes, config):
        super(CNN, self).__init__()
        self.config = config
        layers = []
        in_channels = config["model"]["input_channels"]  # Read input channels (e.g., 3 for RGB)

        # Add convolutional layers dynamically
        for layer in config["model"]["conv_layers"]:
            layers.append(nn.Conv2d(in_channels, layer["filters"], kernel_size=layer["kernel_size"]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=layer["pool_size"], stride=layer["pool_size"]))
            in_channels = layer["filters"]  # Update for next layer
        
        self.features = nn.Sequential(*layers)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 17 * 17, config["model"]["fc_layers"][0]["units"]),
            nn.ReLU(inplace=True),
            nn.Linear(config["model"]["fc_layers"][0]["units"], config["config"]["num_classes"])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
