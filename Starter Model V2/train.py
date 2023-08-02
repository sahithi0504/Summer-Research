import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
import os
import random
import numpy as np
import torch.nn.init as init
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set the seed value
seed_value = 42

# Set the seed for PyTorch
torch.manual_seed(seed_value)

# If you are using numpy as well
np.random.seed(seed_value)

# If you are using random module as well
random.seed(seed_value)

# If you are using GPUs, set the seed for the current GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import CountingModel from model.py
from model import CountingModel

# Variable for telling the program to use a GPU
device = torch.device('cuda')

# Initialize model
model = CountingModel()
model.to(device)

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
        image_key = self.image_keys[idx]
        img_name = os.path.join(self.image_folder, self.data_info[image_key]["file_name"])
        image = Image.open(img_name).convert('L')
        num_dots = self.data_info[image_key]["num_dots"]

        if self.transform:
            image = self.transform(image)

        return image, num_dots

# Configuration
batch_size = 16
print("Batch Size: " + str(batch_size))

learning_rate = 0.0001
print("Learning Rate: " + str(learning_rate))

num_epochs = 30

# Transforms
transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Where the training images and their labels are
image_folder = "C:/Users/Skylar/Desktop/New fake eggs(not overlapping and two diff sizes)"
json_file = "C:/Users/Skylar/Documents/VSCode Projects/Python Projects/eggCounter/labels(no-overlap-2sizes).json"

# Load Dataset
dataset = CustomDataset(image_folder=image_folder, json_file=json_file, transform=transform)
# dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Split into train, validation, and test
dataset_size = len(dataset)
indices = list(range(dataset_size))
split_train = int(0.7 * dataset_size)  # 70% for training
split_val = int(0.85 * dataset_size)   # 15% for validation

# Randomly shuffle indices
np.random.shuffle(indices)

train_indices = indices[:split_train]
val_indices = indices[split_train:split_val]
test_indices = indices[split_val:]

# Create Samplers
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Loss function
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
print(criterion)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
print(optimizer.__class__)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.01)

# Before the training loop
best_val_loss = float('inf')
epochs_without_improvement = 0
max_epochs_without_improvement = 3 # You can change this

# Training Loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0

    for images, num_dots in train_loader:
        images, num_dots = images.to(device), num_dots.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, num_dots.float().unsqueeze(1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    # Reset running loss for validation
    running_val_loss = 0.0

    # Validation phase
    model.eval()
    all_outputs = []
    all_num_dots = []
    with torch.no_grad():
        for images, num_dots in val_loader:
            images, num_dots = images.to(device), num_dots.to(device)
            outputs = model(images)
            loss = criterion(outputs, num_dots.float().unsqueeze(1))
            running_val_loss += loss.item()
            all_outputs.extend(outputs.cpu().numpy())
            all_num_dots.extend(num_dots.cpu().numpy())

    avg_train_loss = running_train_loss / len(train_loader)
    avg_val_loss = running_val_loss / len(val_loader)
    
    # Calculating MAE and R2 Score
    mae = mean_absolute_error(all_num_dots, all_outputs)
    r2 = r2_score(all_num_dots, all_outputs)

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        # Save the model
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'counting_model.ckpt')
        torch.save(model.state_dict(), checkpoint_path)
    else:
        epochs_without_improvement += 1
        # If the validation loss hasn't improved for max_epochs_without_improvement epochs, stop training
        if epochs_without_improvement > max_epochs_without_improvement:
            print("Early stopping")
            break

    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}, MAE: {mae:.8f}, R2 Score: {r2:.8f}")

# Save the model
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_dir, 'counting_model.ckpt')
torch.save(model.state_dict(), checkpoint_path)