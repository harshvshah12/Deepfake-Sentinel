import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), (128, 128, 128))
            
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def train_and_calibrate(epochs=2, batch_size=8, lr=5e-5, device='cpu'):
    dev = torch.device(device)
    print(f'[Trainer] Initiating training loop on device: {dev}')

    real_files = glob.glob('dataset/real/*.jpg') + glob.glob('dataset/real/*.png')
    fake_files = glob.glob('dataset/fake/*.jpg') + glob.glob('dataset/fake/*.png')

    min_samples = min(len(real_files), len(fake_files))
    real_sample = random.sample(real_files, min_samples)
    fake_sample = random.sample(fake_files, min_samples)

    all_files = real_sample + fake_sample
    all_labels = [0.0] * len(real_sample) + [1.0] * len(fake_sample)

    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)

    split = int(0.85 * len(all_files))
    train_files, val_files = all_files[:split], all_files[split:]
    train_labels, val_labels = all_labels[:split], all_labels[split:]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = DeepfakeDataset(train_files, train_labels, transform=train_transform)
    val_ds = DeepfakeDataset(val_files, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    import torchvision.models as models
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    weights_dir = Path('backend/weights/deepfakefusion')
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / 'best_efficientnet_b3_deepfake_detector_399k.pth'

    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=dev)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print('[Trainer] Loaded pre-trained base checkpoint.')

    model.to(dev)

    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(dev), targets.to(dev).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(dev), targets.to(dev).unsqueeze(1)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = correct / max(1, total)
        print(f'[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%')

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'threshold': 0.45,
                'val_accuracy': val_acc
            }, weights_dir / 'local_finetune_NOT_PRODUCTION.pth')
            print('[Trainer] Saved a separate local checkpoint. The production 399k file was not overwritten.')

    print(f'[Trainer] Optimization complete! Best Accuracy: {best_acc*100:.2f}%')

if __name__ == '__main__':
    train_and_calibrate(epochs=2, batch_size=8, lr=5e-5, device='cpu')
