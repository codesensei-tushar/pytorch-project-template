import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms


class ButterflyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # Add this line to check the contents of the CSV
        # self.annotations = pd.read_csv(csv_file, names=["filename", "label"], header=0)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}
        # print(f"Number of classes: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[index, 1]
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(config):
    full_train_dataset = ButterflyDataset(config['config']['train_csv'], config['config']['train_dir'])
    
    train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    test_size = len(full_train_dataset) - train_size

    train_dataset, test_dataset = random_split(full_train_dataset, [train_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['config']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['config']['batch_size'],
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader