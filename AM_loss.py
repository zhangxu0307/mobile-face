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
        self.weight = Parameter(th.Tensor(inputDim, classNum))
        self.weight.data.normal_(-1, 1).renorm_(2,1,1e-5)
        self._reset_params()

    def _reset_params(self):

        xavier_normal(self.weight)

    def forward(self, x):

        out = F.normalize(x, p=2, dim=1)  # x的维度(batchsize, dim_rep)
        normWeight = F.normalize(self.weight, p=2, dim=0)
        out = th.mm(out, normWeight)
        out = out.clamp(-1, 1)

        # w = self.weight
        # ww = w.renorm(2, 1, 1e-5)
        # xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        # wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum
        #
        # cos_theta = x.mm(ww)  # size=(B,Classnum)
        # cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        # cos_theta = cos_theta.clamp(-1, 1)
        # out = cos_theta

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

        y_onehot = th.zeros((x.size()[0], self.classNum)).cuda()
        y_onehot.scatter_(1, y.data.view(-1, 1), self.s * self.m)
        y_onehot = Variable(y_onehot, requires_grad=False)

        # y = th.from_numpy(y)
        # y_onehot = y.data.cpu().numpy()
        # y_onehot = (np.arange(self.classNum) == y_onehot[:, None]).astype(np.float32)
        # y_onehot *= self.s * self.m
        # y_onehot = Variable(th.from_numpy(y_onehot).cuda(), requires_grad=False)

        out = x - y_onehot

        # logpt = F.log_softmax(out)
        # logpt = logpt.gather(1, y)
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        #
        # loss = -1 * (1 - pt) ** self.gamma * logpt
        # loss = loss.mean()

        loss = self.loss(out, y)

        return loss


if __name__ == '__main__':
    pass





