import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as nnfun
import torch.utils.data

class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = nnfun.relu(self.conv1(x))
        x = nnfun.relu(self.conv2(x))
        x = self.conv3(x)
        return x