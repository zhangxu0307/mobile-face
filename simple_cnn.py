import torch.nn as nn
import torch.nn.functional as F
from AM_loss import *

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.BN1 = nn.BatchNorm2d(6)
        self.BN2 = nn.BatchNorm2d(16)

        self.AM = AMLayer(inputDim=256, s=5, m=0.6, classNum=10)

    def forward(self, x, target):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.BN1(out)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.BN2(out)

        out = out.view(out.size(0), -1)
        out = self.AM(out, target)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)


        return out