import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse

from mobile_net import *
from simple_cnn import *
from load_data import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()





def train(model, batchSize, epoch, useCuda=True):

    if useCuda:
        model = model.cuda()

    ceriation = nn.CrossEntropyLoss()
    #AM = AMLayer(inputDim=1280, s=30, m=0.4, classNum=10)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    trainLoader, testLoader = loadMNIST(batchSize=batchSize)

    for i in range(epoch):

        # trainning
        sum_loss = 0

        for batch_idx, (x, target) in enumerate(trainLoader):

            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()

            x, target = Variable(x), Variable(target)
            out = model(x, target)
            # out = AM(out, target)

            loss = ceriation(out, target)
            sum_loss += loss.data[0]

            loss.backward(retain_graph=True)
            optimizer.step()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(i, batch_idx + 1, sum_loss/(batch_idx+1)))

        # testing
        correct_cnt, sum_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(testLoader):
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            if useCuda:
                x, target = x.cuda(), target.cuda()
            out = model(x, target)
            loss = ceriation(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

            # smooth average
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(testLoader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    i, batch_idx + 1, sum_loss/(batch_idx+1), correct_cnt * 1.0 / total_cnt))


if __name__ == '__main__':


    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        net = LeNet()
        #net = MobileNetV2()


    use_cuda = torch.cuda.is_available()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        #cudnn.benchmark = True

    train(model=net, epoch=10, batchSize=256, useCuda=use_cuda)


