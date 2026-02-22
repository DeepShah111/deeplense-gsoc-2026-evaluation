import os
from torch.utils.data import Dataset
from PIL import Image

class DeepLenseDataset(Dataset):
    #Loads Grayscale (1-channel) images for the custom Baseline CNN.
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = {'no_sub': 0, 'cdm': 1, 'vortex': 2}

    def __len__(self): 
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['class'], row['filename'])
        image = Image.open(img_path).convert('L')
        if self.transform: image = self.transform(image)
        return image, self.class_map[row['class']]

class DeepLenseDatasetRGB(Dataset):
    #Loads RGB (3-channel) images for Transfer Learning and ViT models.
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = {'no_sub': 0, 'cdm': 1, 'vortex': 2}

    def __len__(self): 
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['class'], row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.class_map[row['class']]