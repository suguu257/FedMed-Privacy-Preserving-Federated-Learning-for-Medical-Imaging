import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.xray_resnet import get_xray_model
from scripts.dataset import XrayDataset
from scripts.utils import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    all_acc, all_prec, all_rec, all_auc = [], [], [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)  # BCEWithLogitsLoss expects float

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  # [batch_size] logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        acc, prec, rec, auc = compute_metrics(outputs, labels)
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_auc.append(auc)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = sum(all_acc) / len(all_acc)
    epoch_prec = sum(all_prec) / len(all_prec)
    epoch_rec = sum(all_rec) / len(all_rec)
    epoch_auc = sum(all_auc) / len(all_auc)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_auc


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0

    all_acc, all_prec, all_rec, all_auc = [], [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            acc, prec, rec, auc = compute_metrics(outputs, labels)
            all_acc.append(acc)
            all_prec.append(prec)
            all_rec.append(rec)
            all_auc.append(auc)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = sum(all_acc) / len(all_acc)
    epoch_prec = sum(all_prec) / len(all_prec)
    epoch_rec = sum(all_rec) / len(all_rec)
    epoch_auc = sum(all_auc) / len(all_auc)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_auc


def main():
    # Paths for hospital 1
    train_dir = os.path.join("data", "hospital_1", "train")
    val_dir = os.path.join("data", "hospital_1", "val")

    # Create datasets
    train_dataset = XrayDataset(train_dir, train=True)
    val_dataset = XrayDataset(val_dir, train=False)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = get_xray_model(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5  # you can increase later

    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_rec, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_acc, val_prec, val_rec, val_auc = evaluate(
            model, val_loader, criterion
        )

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, "
              f"prec={train_prec:.4f}, rec={train_rec:.4f}, auc={train_auc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, "
              f"prec={val_prec:.4f}, rec={val_rec:.4f}, auc={val_auc:.4f}")

    # Save the trained model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "xray_resnet_hospital1.pth"))
    print("\nModel saved at checkpoints/xray_resnet_hospital1.pth")


if __name__ == "__main__":
    main()
