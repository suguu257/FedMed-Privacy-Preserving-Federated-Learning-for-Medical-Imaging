import os
import argparse

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.xray_resnet import get_xray_model
from scripts.dataset import XrayDataset
from scripts.utils import compute_metrics
from scripts.train_local import train_one_epoch, evaluate  # reuse your existing functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Differential Privacy hyperparameters ----
DP_CLIP_NORM = 1.0         # maximum L2 norm of gradients (clipping bound)
DP_NOISE_MULTIPLIER = 0.1  # how much noise to add; higher = more privacy, less accuracy

def dp_train_one_epoch(model, dataloader, optimizer, criterion,
                       clip_norm=DP_CLIP_NORM, noise_multiplier=DP_NOISE_MULTIPLIER):
    """
    One epoch of DIFFERENTIALLY PRIVATE training:
    - Clip gradients to a fixed L2 norm
    - Add Gaussian noise to gradients
    """
    model.train()
    running_loss = 0.0

    all_acc, all_prec, all_rec, all_auc = [], [], [], []

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  # [batch_size] logits
        loss = criterion(outputs, labels)
        loss.backward()

        # 1) Gradient clipping (limits sensitivity)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        # 2) Add Gaussian noise to each gradient tensor
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_multiplier * clip_norm,
                    size=p.grad.shape,
                    device=p.grad.device,
                )
                p.grad += noise

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Metrics
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

def get_model_parameters(model):
    """Convert model parameters to a list of numpy arrays for Flower."""
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    """Set model parameters from a list of numpy arrays received from Flower."""
    state_dict = model.state_dict()
    new_state_dict = {}

    # parameters is a list of numpy arrays, same order as state_dict.items()
    for (key, _), param in zip(state_dict.items(), parameters):
        new_state_dict[key] = torch.tensor(param)

    model.load_state_dict(new_state_dict, strict=True)


class MedicalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id

        # Paths specific to this hospital
        train_dir = os.path.join("data", f"hospital_{hospital_id}", "train")
        val_dir = os.path.join("data", f"hospital_{hospital_id}", "val")

        # Datasets and loaders
        self.train_dataset = XrayDataset(train_dir, train=True)
        self.val_dataset = XrayDataset(val_dir, train=False)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=16, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=16, shuffle=False, num_workers=0
        )

        # Model, loss, optimizer
        self.model = get_xray_model(num_classes=1).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        print(f"Hospital {hospital_id}:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples:   {len(self.val_dataset)}")

    # Flower NumPyClient interface:

    def get_parameters(self, config):
        """Return current model parameters."""
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Train model on local data."""

        # Receive global parameters from server
        set_model_parameters(self.model, parameters)

        # Local training
        local_epochs = int(config.get("local_epochs", 1))

        for epoch in range(local_epochs):
            train_loss, train_acc, train_prec, train_rec, train_auc = dp_train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion
            )
            print(
                f"[Hospital {self.hospital_id}] "
                f"Local epoch: loss={train_loss:.4f}, acc={train_acc:.4f}"
            )

        # Return updated parameters and the number of training examples
        return get_model_parameters(self.model), len(self.train_dataset), {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
        }

    def evaluate(self, parameters, config):
        """Evaluate model on local validation data."""

        # Use the parameters from the server
        set_model_parameters(self.model, parameters)

        if len(self.val_dataset) == 0:
            print(f"[Hospital {self.hospital_id}] No validation data, skipping eval.")
            return 0.0, 0, {"val_loss": 0.0, "val_acc": 0.0}

        val_loss, val_acc, val_prec, val_rec, val_auc = evaluate(
            self.model, self.val_loader, self.criterion
        )

        print(
            f"[Hospital {self.hospital_id}] "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        # Flower expects: loss, number of examples, metrics dict
        return float(val_loss), len(self.val_dataset), {
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hospital-id",
        type=int,
        required=True,
        help="Hospital ID (1, 2, 3, ...). Determines which data folder to use.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Flower server address",
    )
    args = parser.parse_args()

    # Create Flower client
    client = MedicalClient(hospital_id=args.hospital_id)

    # Start Flower client and connect to server
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
