import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import CountingModel

# Load the trained model
model = CountingModel()
model.load_state_dict(torch.load("C:/Users/Skylar/Documents/VSCode Projects/Python Projects/eggCounter/counting_model.ckpt"))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor()
])

# Load and transform the test image
image_path = "C:/Users/Skylar/Desktop/120-dots-360x360 (3).png"
test_image = Image.open(image_path).convert('RGB')
test_image = transform(test_image).unsqueeze(0)  # Add batch dimension

# Predict the number of dots
with torch.no_grad():
    output = model(test_image)

predicted_count = int(torch.round(output.squeeze()))

print("Predicted Dot Count:", predicted_count)
