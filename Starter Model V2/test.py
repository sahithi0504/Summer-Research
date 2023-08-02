import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import CountingModel

# Check for GPU
device = torch.device('cuda')


# Load the trained model
model = CountingModel()
model.load_state_dict(torch.load("C:/Users/Skylar/Documents/VSCode Projects/Python Projects/eggCounter/counting_model.ckpt"))
model = model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Path to the test image folder
image_folder = "C:/Users/Skylar/Desktop/test images"

# List all image filenames in the folder
image_filenames = os.listdir(image_folder)

# Iterate over the images and predict the number of dots
for filename in image_filenames:
    image_path = os.path.join(image_folder, filename)
    
    # Load the image without converting to RGB
    test_image = Image.open(image_path)
    test_image = transform(test_image).unsqueeze(0)  # Add batch dimension
    
    # Move the input data to the same device as the model
    test_image = test_image.to(device)

    with torch.no_grad():
        output = model(test_image)

    # Apply inverse normalization
    # predicted_count = inverse_normalize(output.squeeze().item())

    # Round the count to the nearest integer
    predicted_count = round(output.squeeze().item())

    print("Image:", filename)
    print("Predicted Dot Count:", predicted_count)
    print()