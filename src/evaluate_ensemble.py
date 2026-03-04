import os
import argparse
import torch
from tqdm import tqdm

from dataset import get_dataloaders, stage_data_locally
from models import ResNetTransfer, ViTChampion, DeepLenseEnsemble
from metrics import save_confusion_matrix, generate_classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the DeepLense Ensemble Model")
    parser.add_argument('--data_dir', type=str, default=".", help='Path to the image directory')
    parser.add_argument('--csv_path', type=str, default="metadata.csv", help='Path to metadata.csv')
    parser.add_argument('--zip_path', type=str, required=True, help='Path to zipped dataset in Drive')
    parser.add_argument('--resnet_weights', type=str, default="weights/transfer_final.pth", help='Path to saved ResNet weights')
    parser.add_argument('--vit_weights', type=str, default="weights/vit_final.pth", help='Path to saved ViT weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for validation')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating Ensemble on device: {device}")

    # 1. Automate Data Staging
    staged_dir = stage_data_locally(args.zip_path)
    if staged_dir:
        args.data_dir = staged_dir
        args.csv_path = os.path.join(staged_dir, os.path.basename(args.csv_path))

    # 2. Load Validation Data (224x224 RGB for the ViT entry point)
    _, val_loader = get_dataloaders(
        csv_path=args.csv_path, 
        base_dir=args.data_dir, 
        mode='RGB', 
        image_size=224, 
        batch_size=args.batch_size
    )

    # 3. Initialize Base Models
    print("Loading base models...")
    resnet = ResNetTransfer(num_classes=3)
    vit = ViTChampion(num_classes=3)

    # 4. Load Saved Weights
    resnet.load_state_dict(torch.load(args.resnet_weights, map_location=device))
    vit.load_state_dict(torch.load(args.vit_weights, map_location=device))

    # 5. Initialize the Ensemble
    print("Fusing models into DeepLenseEnsemble...")
    ensemble_model = DeepLenseEnsemble(resnet_model=resnet, vit_model=vit)
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()

    # 6. Evaluation Loop
    print("\nRunning Final Ensemble Validation...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating Ensemble")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # The ensemble handles downsampling for the ResNet internally!
            ensemble_probs = ensemble_model(images)
            _, predicted = torch.max(ensemble_probs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. Generate Metrics
    classes = ['No Sub', 'CDM', 'Vortex']
    generate_classification_report(all_labels, all_preds, classes)
    
    os.makedirs("assets", exist_ok=True)
    cm_path = os.path.join("assets", "ensemble_confusion_matrix.png")
    save_confusion_matrix(all_labels, all_preds, classes=classes, save_path=cm_path, title='ENSEMBLE (ResNet + ViT) Evaluation')
    print(f"\n✅ Evaluation complete. Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()