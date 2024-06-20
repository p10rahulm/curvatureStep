import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, 
                 image_width=96, 
                 num_channels=3):
        super(SimpleCNN, self).__init__()
        self.num_channels=num_channels
        self.image_width=image_width
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (image_width // 4) * (image_width // 4), 512)  # Adjust the input size of the first fully connected layer
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (self.image_width // 4) * (self.image_width // 4))  # Adjust the view to match the input size of the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
