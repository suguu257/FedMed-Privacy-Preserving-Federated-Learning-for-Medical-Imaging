import flwr as fl
import torch
import torch.nn as nn
from torchvision import models
from opacus.validators import ModuleValidator
from flwr.common import ndarrays_to_parameters

# Define the Model (Must match Client exactly)
def get_model():
    # Load standard ResNet18
    model = models.resnet18(pretrained=True)
    
    # Freeze ReLU (Opacus requirement)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            
    # Adjust output layer for 2 classes (Pneumonia vs Normal)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Convert BatchNorm to GroupNorm (Opacus requirement)
    model = ModuleValidator.fix(model)
    return model

# Start the Server
if __name__ == "__main__":
    # Initialize the global model parameters
    model = get_model()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    # Define the Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # Sample 100% of available clients for training
        fraction_evaluate=1.0,        # Sample 100% of available clients for evaluation
        min_fit_clients=2,            # Never start training until 2 clients are connected
        min_evaluate_clients=2,       # Never start evaluation until 2 clients are connected
        min_available_clients=2,      # Wait for 2 clients before starting the round
        initial_parameters=parameters, # Pass the initial weights here!
    )

    print("🚀 Server is starting... Waiting for 2 Hospitals to connect.")
    print("📋 Configuration:")
    print("   - Federated Rounds: 10")
    print("   - Epochs per round: 5")
    print("   - Total training epochs: 50")

    # Start Server with 10 ROUNDS
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )