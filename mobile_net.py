'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import numpy as np
from AM_loss import *
import cv2
from net_sphere import *


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.relu = th.nn.PReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10575, feature=False):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10

        self.feature = feature

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.linear = nn.Linear(1280, num_classes)

        self.relu = th.nn.PReLU()

        self.AM = AMLayer(inputDim=1280, classNum=num_classes)
        # self.sphereLayer = AngleLinear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)

        if self.feature:
            return out

        out = self.AM(out)
        return out

    def getRep(self, x):

        x = x.astype(np.float64)
        x = (x/255 - 0.5) / 0.5
        x = np.transpose(x, [2, 0, 1])
        x = np.expand_dims(x, axis=0)
        x = Variable(torch.from_numpy(x).float(), volatile=True).cuda()

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)

        return out.data.cpu().numpy()


if __name__ == '__main__':

    model = MobileNetV2()
