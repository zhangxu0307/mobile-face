import torch.nn as nn
import torch.nn.functional as F
from AM_loss import *
import numpy as np
from torch.nn.init import xavier_normal, kaiming_normal

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 64)
        self.fc2 = nn.Linear(64, 2)
        self.BN1 = nn.BatchNorm2d(6)
        self.BN2 = nn.BatchNorm2d(16)

        self.AM = AMLayer(inputDim=64, classNum=10)

    def forward(self, x, target):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.BN1(out)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.BN2(out)

        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        #out = F.relu(self.fc2(out))
        out = self.AM(out)

        return out

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10575):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        #self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.AM = AMLayer(512*block.expansion, s=30, m=0.45, classNum=num_classes)

        self.linear = nn.Linear(32768, 1024)
        self.AM = AMLayer(512*block.expansion, classNum=num_classes)

        self._reset_parameters()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _reset_parameters(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                kaiming_normal(m.weight)

            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        out = self.AM(out)
        return out

    def getRep(self, x):

        x = x.astype(np.float64)
        x = (x/255-0.5) / 0.5
        x = np.transpose(x, [2, 0, 1])
        x = np.expand_dims(x, axis=0)
        x = Variable(torch.from_numpy(x).float(), volatile=True).cuda()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)

        return out.data.cpu().numpy()


def ResNet18(classNum):
    return ResNet(BasicBlock, [2,2,2,2], classNum)

def ResNet34(classNum):
    return ResNet(BasicBlock, [3,4,6,3], classNum)

def ResNet50(classNum):
    return ResNet(Bottleneck, [3,4,6,3], classNum)

def ResNet101(classNum):
    return ResNet(Bottleneck, [3,4,23,3], classNum)

def ResNet152(classNum):
    return ResNet(Bottleneck, [3,8,36,3], classNum)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()