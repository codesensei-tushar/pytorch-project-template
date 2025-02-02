import torch



class Config:
    # Data
    train_dir = "train/"
    test_dir = "test/"
    train_csv = "Training_csv/train.csv"  # Assuming CSV has 'image' and 'class' columns
    test_csv = "Testing_csv/test.csv"
    
    # Model
    img_size = 150
    num_classes = 75
    
    # Training
    batch_size = 32
    epochs = 10
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")