# models/simpleDQN.py

import torch.nn as nn
import torch

class SimpleDQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(obs_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
