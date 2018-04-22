import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.init import kaiming_normal, xavier_normal

class AMLayer(nn.Module):

    def __init__(self, inputDim, classNum):

        super().__init__()

        self.classNum = classNum
        self.weight = Parameter(th.randn(inputDim, classNum).cuda())
        self._reset_params()

    def _reset_params(self):

        kaiming_normal(self.weight)

    def forward(self, x):

        out = F.normalize(x, p=2, dim=1)  # x的维度(batchsize, dim_rep)
        normWeight = F.normalize(self.weight, p=2, dim=0)
        out = th.mm(out, normWeight)

        return out

class AMLoss(nn.Module):

    def __init__(self, s, m, classNum):

        super().__init__()
        self.s = s
        self.m = m
        self.classNum = classNum
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):

        x = self.s * x

        y_onehot = th.zeros((x.size()[0], self.classNum)).cuda().scatter_(1, y.data.view(-1, 1), self.s * self.m)
        y_onehot = Variable(y_onehot, requires_grad=False)

        # y = th.from_numpy(y)
        # y_onehot = y.data.cpu().numpy()
        # y_onehot = (np.arange(self.classNum) == y_onehot[:, None]).astype(np.float32)
        # y_onehot *= self.s * self.m
        # y_onehot = Variable(th.from_numpy(y_onehot).cuda(), requires_grad=False)

        out = x - y_onehot

        loss = self.loss(out, y)

        return loss


if __name__ == '__main__':
    pass





