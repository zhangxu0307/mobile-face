import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

class AMLayer(nn.Module):

    def __init__(self, inputDim, classNum, s, m):
        super().__init__()
        self.s = s
        self.m = m
        self.s_m = s*m
        self.classNum = classNum
        self.weight = Parameter(th.randn(inputDim, classNum).cuda())

    def forward(self, x, y):

        out = F.normalize(x, p=2)
        out = out*self.s
        normWeight = F.normalize(self.weight, p=2)
        out = th.mm(out, normWeight)


        y_onehot = y.data.cpu().numpy()
        y_onehot = (np.arange(self.classNum) == y_onehot[:, None]).astype(np.float32)
        y_onehot *= self.s_m
        y_onehot = Variable(th.from_numpy(y_onehot).cuda(), requires_grad=False)

        out = out - y_onehot

        return out

if __name__ == '__main__':

    pass





