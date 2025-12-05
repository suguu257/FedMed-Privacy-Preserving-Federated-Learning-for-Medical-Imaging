import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import flwr as fl
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = "./chest_xray/train"  
BATCH_SIZE = 32  # Increased from 16 for better gradient estimates

# --- 1. LOAD REAL DATA WITH AUGMENTATION ---
def get_data_loaders():
    # Training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)
    except FileNotFoundError:
        print(f"ERROR: Could not find data at {DATA_ROOT}")
        print("Make sure you extracted the chest_xray folder into the same directory as client.py")
        sys.exit(1)

    # Split data: 80% for training, 20% for testing (local validation)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Apply test transform to test dataset
    test_dataset.dataset = datasets.ImageFolder(root=DATA_ROOT, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

# --- MONKEY PATCH FOR OPACUS COMPATIBILITY ---
from torchvision.models.resnet import BasicBlock

def opacus_friendly_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity 
    out = self.relu(out)
    return out

BasicBlock.forward = opacus_friendly_forward
# ---------------------------------------------

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Model
def get_model():
    model = models.resnet18(pretrained=True)
    
    # Fix ReLU for privacy engine
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            
    # Adjust final layer for 2 classes (Normal vs Pneumonia)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Convert Batch Normalization to Group Normalization for Opacus
    model = ModuleValidator.fix(model)
    return model

# Flower Client Class
class SecureFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = get_model().to(device)
        
        # --- LOAD REAL DATA ---
        print(f"🏥 Hospital {client_id}: Loading real chest X-rays...")
        self.train_loader, self.test_loader = get_data_loaders()
        
        self.criterion = nn.CrossEntropyLoss()
        
        # --- OPTIMIZER ---
        # Using Adam with slightly higher learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)

        # --- PRIVACY ENGINE ---
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=0.3,  # Increased for better privacy-utility tradeoff
            max_grad_norm=2.0,     # Increased for more gradient flow
        )
        print(f"🔒 Hospital {client_id}: Privacy Engine Active!")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        state_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in state_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.train()
        epoch_losses = []
        
        # 5 EPOCHS PER ROUND
        for epoch in range(5):
            running_loss = 0.0
            batch_count = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = running_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            print(f"  📈 Hospital {self.client_id} - Epoch {epoch+1}/5: Loss={avg_epoch_loss:.4f}")
        
        # Update learning rate
        self.scheduler.step()
        
        avg_round_loss = np.mean(epoch_losses)
        print(f"✅ Hospital {self.client_id} finished training round. Avg Loss: {avg_round_loss:.4f}")
        return self.get_parameters(config={}), len(self.train_loader), {"loss": float(avg_round_loss)}

    def evaluate(self, parameters, config):
        state_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in state_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        
        # Evaluate on the TEST loader (unseen data)
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss_sum += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Avoid division by zero if loader is empty
        if len(self.test_loader) == 0:
            avg_loss = 0
            accuracy = 0
        else:
            avg_loss = loss_sum / len(self.test_loader)
            accuracy = correct / total
        
        print(f"📊 Hospital {self.client_id} Evaluation: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f} ({correct}/{total})")
        return float(avg_loss), len(self.test_loader), {"accuracy": float(accuracy)}

# Start Client
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <hospital_id>")
        sys.exit(1)
        
    client_id = sys.argv[1]
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=SecureFlowerClient(client_id)
    )