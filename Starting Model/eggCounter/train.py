import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
import os
import torch.cuda as cuda
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Import CountingModel from model.py
from model import CountingModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset
class CustomDataset(data.Dataset):
    def __init__(self, image_folder, json_file, transform=None):
        with open(json_file) as f:
            self.data_info = json.load(f)
        self.image_keys = list(self.data_info.keys())
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Access the data using the keys
        image_key = self.image_keys[idx]
        img_name = os.path.join(self.image_folder, self.data_info[image_key]["file_name"])
        image = Image.open(img_name).convert('RGB')
        num_dots = self.data_info[image_key]["num_dots"]

        if self.transform:
            image = self.transform(image)

        return image, num_dots



# Configuration
image_folder = "C:/Users/Skylar/Desktop/Fake eggs"
json_file = "C:/Users/Skylar/Documents/VSCode Projects/Python Projects/eggCounter/labels.json"
batch_size = 32
learning_rate = 0.008
num_epochs = 10

# Transformations
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

# Load Dataset
dataset = CustomDataset(image_folder=image_folder, json_file=json_file, transform=transform)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss and optimizer
model = CountingModel()
model.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, num_dots in dataloader:
        images, num_dots = images.to(device), num_dots.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, num_dots.float().unsqueeze(1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the model checkpoint
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_dir, 'counting_model.ckpt')
torch.save(model.state_dict(), checkpoint_path)
