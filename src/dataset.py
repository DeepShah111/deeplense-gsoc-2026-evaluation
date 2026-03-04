import os
import shutil
import zipfile
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

def stage_data_locally(drive_zip_path, local_extract_dir="/content/local_dataset"):
    """
    Automates data staging from Google Drive to Colab's local disk.
    Prevents PyTorch DataLoader ConnectionAbortedError by bypassing Drive I/O.
    """
    if drive_zip_path and os.path.exists(drive_zip_path):
        if not os.path.exists(local_extract_dir):
            print(f"📦 Staging data locally to {local_extract_dir} for high-speed I/O...")
            os.makedirs(local_extract_dir, exist_ok=True)
            
            local_zip = "/content/temp_dataset.zip"
            shutil.copy2(drive_zip_path, local_zip)
            
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(local_extract_dir)
                
            os.remove(local_zip)
            print("✅ Data staging complete.")
        return local_extract_dir
    return None

class DeepLenseDataset(Dataset):
    """
    Unified PyTorch Dataset for DeepLense gravitational lensing images.
    Supports both 'L' (Grayscale/1-channel) and 'RGB' (3-channel) modes.
    """
    def __init__(self, dataframe, root_dir, transform=None, mode='L'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode 
        self.class_map = {'no_sub': 0, 'cdm': 1, 'vortex': 2}

    def __len__(self): 
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['class'], row['filename'])
        
        image = Image.open(img_path).convert(self.mode)
        
        if self.transform: 
            image = self.transform(image)
            
        return image, self.class_map[row['class']]

def get_dataloaders(csv_path, base_dir, mode='L', image_size=64, batch_size=32, test_size=0.2):
    """
    Handles the train/test split and returns optimized PyTorch DataLoaders.
    Upgraded with dynamic sizing and standard ImageNet normalization.
    """
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['class'])

    # Apply ImageNet normalization for RGB transfer learning, basic normalization for Grayscale
    if mode == 'RGB':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = DeepLenseDataset(train_df, base_dir, transform=data_transform, mode=mode)
    val_dataset = DeepLenseDataset(val_df, base_dir, transform=data_transform, mode=mode)

    # num_workers=2 is safe now because we will run off the fast local Colab disk
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader