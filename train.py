import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse

from mobile_net import *
from model import *
from load_data import *
import time




def train(model, trainLoader, lr, epoch, modelPath, valid=False):

    # 启动cuda
    useCuda = torch.cuda.is_available()
    if useCuda:
        model = model.cuda()

    # loss, opt和数据loader
    ceriation = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)


    for i in range(epoch):

        start = time.time()

        sum_loss = 0

        for batch_idx, (x, target) in enumerate(trainLoader):

            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x, target)

            loss = ceriation(out, target)
            sum_loss += loss.data[0]

            loss.backward(retain_graph=True)
            optimizer.step()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(i+1, batch_idx + 1, sum_loss/(batch_idx+1)))

        if valid:

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

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(testLoader):
                    print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                        i+1, batch_idx + 1, sum_loss/(batch_idx+1), correct_cnt * 1.0 / total_cnt))

        end = time.time()
        print('==>>> epoch: {} running time: {:.2f}s'.format(i+1, end-start))

    torch.save(model, modelPath)


if __name__ == '__main__':


    rootPath = "data/CASIA-WebFace/"
    modelPath = "model_file/resnet101_AM_webface.pt"
    batchSize = 128
    epoch = 20
    lr = 0.1

    print("==>load data...")
    #trainLoader, testLoader = loadCIFAR10(batchSize=batchSize)
    trainLoader, classNum = loadWebface(rootPath, batchSize, inputsize=128)
    print("==>load data finished!")

    print('==> Building model..')
    # net = th.load(modelPath)
    # net = LeNet()
    # net = MobileNetV2()
    net = ResNet50(classNum)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    net = torch.nn.DataParallel(net, device_ids=range(8))
    torch.cuda.synchronize()

    train(model=net, trainLoader=trainLoader, lr=lr, epoch=epoch, modelPath=modelPath, valid=False)

