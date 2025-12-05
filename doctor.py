import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import random

# 1. Define the same Brain structure (Must match the saved file)
class SimpleMedicalAI(nn.Module):
    def __init__(self):
        super(SimpleMedicalAI, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # Note: If your first model was different, these numbers might need adjusting.
        # This matches the 'brain.py' structure I gave you in Step 2.
        self.fc1 = nn.Linear(32 * 31 * 31, 2) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# 2. Load the Brain
model = SimpleMedicalAI()
try:
    model.load_state_dict(torch.load("pneumonia_model.pth"))
    model.eval() # Set to "Evaluation Mode" (Turn off learning)
    print("Brain loaded successfully!")
except:
    print("Error: Could not find pneumonia_model.pth. Did you run brain.py?")
    exit()

# 3. Pick a Random Patient
test_dir = "./dataset/test"
categories = ["NORMAL", "PNEUMONIA"]

# Pick a random category (Normal or Sick)
true_condition = random.choice(categories)
folder_path = os.path.join(test_dir, true_condition)

# Pick a random image file
patient_xray = random.choice(os.listdir(folder_path))
image_path = os.path.join(folder_path, patient_xray)

print(f"\nExamining Patient: {patient_xray}")
print(f"True Condition:  {true_condition}")

# 4. Process the Image (Same resize rules as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB") # Open image
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 64, 64)

# 5. The Diagnosis
with torch.no_grad():
    output = model(image_tensor)
    # The output is two numbers [Confidence_Normal, Confidence_Pneumonia]
    # We want the higher one.
    _, predicted = torch.max(output, 1)
    
    diagnosis = categories[predicted.item()]

print(f"AI Diagnosis:    {diagnosis}")

if diagnosis == true_condition:
    print("Result: ✅ CORRECT")
else:
    print("Result: ❌ INCORRECT")