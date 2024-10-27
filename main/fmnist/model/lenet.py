import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .binarization import *


class LeNet_5_Masked(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = MaskedConv2d(1, 20, (5, 5), 1)
        self.conv2 = MaskedConv2d(20, 50, (5, 5), 1)
        self.fc3 = MaskedMLP(4 * 4 * 50, 500)
        self.fc4 = MaskedMLP(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.fc3(x.view(-1, 4 * 4 * 50)))
        
        return self.fc4(x)

