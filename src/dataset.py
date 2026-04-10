import os
import shutil
import zipfile
import multiprocessing
from typing import Optional, Callable, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2  # [GSOC UPGRADE 1] Needed for MixUp/CutMix
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate # [GSOC UPGRADE 1]


# ─────────────────────────────────────────────────────────────────────────────
# DATA STAGING
# ─────────────────────────────────────────────────────────────────────────────

def stage_data_locally(
    drive_zip_path: Optional[str],
    local_extract_dir: str = "/content/local_dataset",
) -> Optional[str]:
    """
    Automates data staging from Google Drive to Colab's local disk.
    Prevents PyTorch DataLoader ConnectionAbortedError by bypassing Drive I/O.

    FIX: Added an environment guard so this function fails gracefully when
    called outside of Google Colab (e.g. local dev machine or CI).
    The hardcoded /content/ paths are Colab-specific; on any other system
    the function now prints a warning and returns None instead of crashing.

    Args:
        drive_zip_path (str | None): Full path to the .zip file on Google Drive.
        local_extract_dir (str): Target directory on Colab's fast local disk.

    Returns:
        str | None: Path to the extracted local directory, or None if skipped.
    """
    # ── Guard: skip silently outside Colab ───────────────────────────────
    if not drive_zip_path:
        print("⚠️  No zip path provided. Skipping local staging.")
        return None

    if not os.path.exists(drive_zip_path):
        print(f"⚠️  Zip file not found at: {drive_zip_path}. Skipping local staging.")
        return None

    # ── Already staged ────────────────────────────────────────────────────
    if os.path.exists(local_extract_dir):
        print(f"✅ Local staging already complete at: {local_extract_dir}")
        return local_extract_dir

    # ── Stage ─────────────────────────────────────────────────────────────
    print(f"📦 Staging data locally to '{local_extract_dir}' for high-speed I/O...")
    os.makedirs(local_extract_dir, exist_ok=True)

    # Use a temp path inside the same directory tree so it's always writable,
    # even outside /content/ (handles non-Colab environments gracefully).
    local_zip = os.path.join(os.path.dirname(local_extract_dir), "_temp_dataset.zip")

    try:
        shutil.copy2(drive_zip_path, local_zip)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(local_extract_dir)
    finally:
        # Always clean up the temp zip, even if extraction fails
        if os.path.exists(local_zip):
            os.remove(local_zip)

    print("✅ Data staging complete.")
    return local_extract_dir


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DeepLenseDataset(Dataset):
    """
    Unified PyTorch Dataset for DeepLense gravitational lensing images.

    Supports both 'L' (Grayscale / 1-channel) and 'RGB' (3-channel) modes.
    The class label is derived from the 'class' column of the metadata CSV.

    Class Mapping:
        no_sub  → 0  (Smooth lens, no dark matter substructure)
        cdm     → 1  (Cold Dark Matter substructure)
        vortex  → 2  (Vortex / quantum condensate dark matter)

    Args:
        dataframe (pd.DataFrame): Subset dataframe (train or val split).
        root_dir (str): Base directory containing class-named subdirectories.
        transform (callable, optional): Torchvision transform pipeline.
        mode (str): PIL image mode — 'L' for grayscale, 'RGB' for 3-channel.
    """

    CLASS_MAP   = {'no_sub': 0, 'cdm': 1, 'vortex': 2}
    CLASS_NAMES = ['no_sub', 'cdm', 'vortex']

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        transform: Optional[Callable] = None,
        mode: str = 'L',
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir  = root_dir
        self.transform = transform
        self.mode      = mode

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row      = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['class'], row['filename'])

        image = Image.open(img_path).convert(self.mode)

        if self.transform:
            image = self.transform(image)

        label = self.CLASS_MAP[row['class']]
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORM FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform(
    mode: str = 'RGB',
    image_size: int = 224,
    augment: bool = True,
) -> transforms.Compose:
    """
    Builds the training transform pipeline.
    """
    if mode == 'RGB':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if augment:
        aug_ops = [
            transforms.RandomHorizontalFlip(p=0.5),               # No preferred axis
            transforms.RandomVerticalFlip(p=0.5),                  # No preferred axis
            transforms.RandomRotation(degrees=360, fill=0),        # Full rotational symmetry
            transforms.ColorJitter(brightness=0.2),                # Source brightness variation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # PSF smearing
        ]
        pipeline = aug_ops + [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        pipeline = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(pipeline)


def get_val_transform(
    mode: str = 'RGB',
    image_size: int = 224,
) -> transforms.Compose:
    """
    Builds the validation/test transform pipeline.
    """
    if mode == 'RGB':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

# ─────────────────────────────────────────────────────────────────────────────
# [GSOC UPGRADE 1] MIXUP & CUTMIX COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_mixup_cutmix_collate(num_classes: int = 3):
    """
    Returns a collate function that randomly applies MixUp or CutMix to batches.
    Crucial for regularizing the Vision Transformer (ViT) on small datasets.
    """
    mixup = v2.MixUp(alpha=0.2, num_classes=num_classes)
    cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
    
    choice = v2.RandomChoice([mixup, cutmix])
    
    def collate_fn(batch):
        images, labels = default_collate(batch)
        return choice(images, labels)
        
    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    csv_path: str,
    base_dir: str,
    mode: str = 'RGB',
    image_size: int = 224,
    batch_size: int = 32,
    augment: bool = True,
    worker_init_fn: Optional[Callable] = None,
    generator: Optional[torch.Generator] = None,
    apply_mixup: bool = False, # [GSOC UPGRADE 1] Trigger for ViT
):
    """
    Builds stratified train/val/test DataLoaders.
    """
    df = pd.read_csv(csv_path)

    print("\n📊 Dataset Class Distribution:")
    dist  = df['class'].value_counts().sort_index()
    total = len(df)
    for cls, count in dist.items():
        print(f"   {cls:<10} : {count:>5} samples  ({100 * count / total:.1f}%)")
    print(f"   {'TOTAL':<10} : {total:>5} samples\n")

    # ── [GSOC UPGRADE 2] STRICT TRAIN / VAL / TEST SPLIT ─────────────────
    # 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30, 
        random_state=42,
        stratify=df['class'],
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50, 
        random_state=42,
        stratify=temp_df['class'],
    )
    print(f"✅ Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Transforms ───────────────────────────────────────────────────────
    train_transform = get_train_transform(mode=mode, image_size=image_size, augment=augment)
    val_transform   = get_val_transform(mode=mode, image_size=image_size)

    # ── Dataset objects ──────────────────────────────────────────────────
    train_dataset = DeepLenseDataset(train_df, base_dir, transform=train_transform, mode=mode)
    val_dataset   = DeepLenseDataset(val_df,   base_dir, transform=val_transform,   mode=mode)
    test_dataset  = DeepLenseDataset(test_df,  base_dir, transform=val_transform,   mode=mode) # [GSOC UPGRADE 2]

    # ── [GSOC UPGRADE 3] HARDWARE-AWARE DATALOADERS ──────────────────────
    hw_num_workers = min(4, multiprocessing.cpu_count()) if torch.cuda.is_available() else 0
    hw_pin_memory = torch.cuda.is_available()

    collate_fn = get_mixup_cutmix_collate(num_classes=3) if (augment and apply_mixup) else None

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size      = batch_size,
        shuffle         = True,
        num_workers     = hw_num_workers,
        pin_memory      = hw_pin_memory,
        drop_last       = True,            
        worker_init_fn  = worker_init_fn,  
        generator       = generator,       
        collate_fn      = collate_fn       # [GSOC UPGRADE 1]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size      = batch_size,
        shuffle         = False,           
        num_workers     = hw_num_workers,
        pin_memory      = hw_pin_memory,
        worker_init_fn  = worker_init_fn,  
    )

    test_loader = DataLoader(              # [GSOC UPGRADE 2]
        test_dataset,
        batch_size      = batch_size,
        shuffle         = False,           
        num_workers     = hw_num_workers,
        pin_memory      = hw_pin_memory,
        worker_init_fn  = worker_init_fn,  
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df