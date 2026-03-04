import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders, stage_data_locally
from models import ResNetBaseline, ResNetTransfer, ViTChampion
from metrics import save_confusion_matrix, generate_classification_report

def set_seed(seed=42):
    """Ensures scientific reproducibility (Strict ML4SCI GSoC requirement)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="DeepLense GSoC 2026 Training Pipeline")
    parser.add_argument('--model_name', type=str, default='baseline', choices=['baseline', 'transfer', 'vit'], 
                        help='Which model to train')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the image directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata.csv')
    parser.add_argument('--zip_path', type=str, default=None, help='(Optional) Path to zipped dataset in Drive for auto-extraction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='weights', help='Directory to save model weights')
    return parser.parse_args()

def main():
    set_seed(42) # Lock the random state for deterministic mentor evaluations
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Executing on device: {device}")

    # 1. Automate Data Staging (Fixes the Colab Drive Crash)
    if args.zip_path:
        staged_dir = stage_data_locally(args.zip_path)
        if staged_dir:
            # Override the data_dir and csv_path to use the new fast local storage
            args.data_dir = staged_dir
            args.csv_path = os.path.join(staged_dir, os.path.basename(args.csv_path))

    # 2. Dynamic Input Configuration for different models
    if args.model_name == 'baseline':
        mode, image_size = 'L', 64
        model = ResNetBaseline(num_classes=3)
    elif args.model_name == 'transfer':
        mode, image_size = 'RGB', 224 
        model = ResNetTransfer(num_classes=3)
    elif args.model_name == 'vit':
        mode, image_size = 'RGB', 224
        model = ViTChampion(num_classes=3)
    
    train_loader, val_loader = get_dataloaders(
        csv_path=args.csv_path, 
        base_dir=args.data_dir, 
        mode=mode, 
        image_size=image_size,
        batch_size=args.batch_size
    )

    model = model.to(device)

    # 3. Setup Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop
    print(f"Starting training for {args.model_name.upper()} model over {args.epochs} epochs...")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # 5. Save Weights
    save_path = os.path.join(args.save_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    # 6. Validation & Metrics
    print("\nRunning Final Validation...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    classes = ['No Sub', 'CDM', 'Vortex']
    generate_classification_report(all_labels, all_preds, classes)
    
    cm_path = os.path.join("assets", f"{args.model_name}_confusion_matrix.png")
    save_confusion_matrix(all_labels, all_preds, classes=classes, save_path=cm_path, title=f'{args.model_name.upper()} Model Evaluation')

if __name__ == "__main__":
    main()