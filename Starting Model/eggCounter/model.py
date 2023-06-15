# Model Definition
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition
class CountingModel(nn.Module):
    def __init__(self):
        super(CountingModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)  # Additional convolutional layer

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 484, 8)  # Adjusted input size
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        # Apply convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))  # New convolutional layer

        # Determine the output size after pooling
        _, _, height, width = x.size()
        output_size = height * width * 128

        # Flatten the tensor
        x = x.view(-1, output_size)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
