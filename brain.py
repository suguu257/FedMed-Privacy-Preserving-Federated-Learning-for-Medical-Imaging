import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# --- PART 1: PREPARING THE DATA ---
# We need to resize all images to 64x64 pixels so the AI can read them easily.
# We also convert them to "Tensors" (which is just a math format for images).
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load the images from your folder
# ImageFolder is a smart tool that knows "NORMAL" is class 0 and "PNEUMONIA" is class 1
print("Loading data... this might take a minute.")
dataset = torchvision.datasets.ImageFolder(root='./dataset/train', transform=transform)

# The DataLoader shuffles the images and gives them to the AI in batches of 16
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


# --- PART 2: BUILDING THE BRAIN (The Model) ---
class SimpleMedicalAI(nn.Module):
    def __init__(self):
        super(SimpleMedicalAI, self).__init__()
        
        # Layer 1: The "Eye"
        # It looks at the image (3 color channels: Red, Green, Blue)
        # It creates 32 different filters (looking for edges, shadows, shapes)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        
        # Layer 2: The "Focus" (Pooling)
        # This shrinks the image information to keep only the important parts
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: The "Decision Maker" (Fully Connected)
        # After looking at the image, we flatten the math into a straight line
        # and ask it to pick between 2 options: Normal or Pneumonia
        # (32 filters * 31 * 31 image size after shrinking)
        self.fc1 = nn.Linear(32 * 31 * 31, 2) 

    def forward(self, x):
        # 1. Look at image with conv1
        x = self.conv1(x)
        # 2. Activate neurons (ReLU makes negatives zero, keeping only "active" signals)
        x = torch.relu(x)
        # 3. Simplify with pool
        x = self.pool(x)
        # 4. Flatten the image into a 1D line
        x = torch.flatten(x, 1)
        # 5. Make a decision
        x = self.fc1(x)
        return x

# Create the brain
model = SimpleMedicalAI()


# --- PART 3: TRAINING (The Classroom) ---
# Loss function: How we calculate how "wrong" the AI is
criterion = nn.CrossEntropyLoss()

# Optimizer: The "Teacher" that updates the brain to reduce errors
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training... (Press Ctrl+C to stop)")

# Loop over the dataset 5 times (5 epochs)
for epoch in range(5): 
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients (reset the teacher for this turn)
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(inputs)           # The AI guesses
        loss = criterion(outputs, labels) # We calculate the error
        loss.backward()                   # We figure out which neurons were wrong
        optimizer.step()                  # The teacher corrects the neurons

        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1} finished. Error Level: {running_loss / len(train_loader)}")

print("Training Finished! Your AI has learned.")

# Save the brain so we can use it later
torch.save(model.state_dict(), "pneumonia_model.pth")