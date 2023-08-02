# Model Definition
import torch
import torch.nn as nn

class CountingModel(nn.Module):
    def __init__(self):
        super(CountingModel, self).__init__()

        # Define the sequential model
        self.model = nn.Sequential(

            # First convolutional block
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second convolutional block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Third convolutional block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Flatten the output tensor
            nn.Flatten(),
            
            # Fully connected layer
            nn.Linear(64 * 45 * 45, 1),

            # Activation function to ensure non-negative output
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

# # Create an instance of the model
# model = CountingModel()

# # Generate a random input tensor
# input_tensor = torch.randn(1, 1, 360, 360)  # Assuming input images of size 360x360 with 3 channels

# # Forward pass
# output = model(input_tensor)
# print(output.shape)  # Check the shape of the output tensor