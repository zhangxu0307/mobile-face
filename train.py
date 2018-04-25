import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mobile_net import *
from resnet import *
from load_data import *
import time
from net_sphere import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def train(model, trainLoader, validLoader, lr, epoch, modelPath, valid=False, checkPoint=10, savePoint=500):

    # 启动cuda
    useCuda = torch.cuda.is_available()
    if useCuda:
        model = model.cuda()

    # loss, opt
    ceriation = AMLoss(s=30, m=0.5, classNum=classNum)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # ceriation = AngleLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    step = 0 # 总迭代次数

    for i in range(epoch):

        trainSumLoss = 0

        for batch_idx, (x, target) in enumerate(trainLoader):

            start = time.time()

            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)

            out = model(x)

            trainLoss = ceriation(out, target)
            trainSumLoss += trainLoss.data[0]

            trainLoss.backward()
            optimizer.step()

            end = time.time()
            batchTime = end-start

            step += 1
            # checkpoint打印一个log
            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>>epoch : {}, batch index: {}, train loss: {:.6f}, step: {}, progress: [{}/{} ({:.0f}%)], running time: {:.2f}s'
                      .format(i+1, batch_idx + 1, trainSumLoss/checkPoint, step,  batch_idx * len(x), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader),  batchTime))
                trainSumLoss = 0

            # 每1000次迭代save一次模型

            if (step+1) % savePoint == 0:
                torch.save(model.state_dict(), modelPath)
                print("==>>>model save finished!")

            # # 每100个batch 做一次valid
            # if (step+1) % 1000 == 0:
            #
            #     correct_cnt, sum_loss = 0, 0
            #     total_cnt = 0
            #
            #     for index, (x, target) in enumerate(validLoader):
            #         x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            #         if useCuda:
            #             x, target = x.cuda(), target.cuda()
            #         out = model(x, target)
            #
            #         validLoss = ceriation(out, target)
            #         _, pred_label = torch.max(out.data, 1)
            #         total_cnt += x.data.size()[0]
            #         correct_cnt += (pred_label == target.data).sum()
            #
            #     print('==>>>acc: {:.3f}'.format(correct_cnt * 1.0 / total_cnt))


if __name__ == '__main__':

    rootPath = "data/webface_detect/"
    modelPath = "model_file/mobilenetv2_webface_align_m05.pt"
    batchSize = 256
    epoch = 10
    lr = 0.001
    inputSize = (112, 96)
    checkPoint = 10

    print("==> load data...")
    # trainLoader, testLoader = loadCIFAR10(batchSize=batchSize)
    trainLoader, classNum = loadWebface(rootPath, batchSize, inputsize=inputSize)
    print("==> load data finished!")

    print('==> Building model..')
    print("==> model path:", modelPath)
    # net = LeNet()
    net = MobileNetV2(classNum)
    # net = ResNet34(classNum)
    # net = sphere20a()
    net.load_state_dict(th.load(modelPath))
    # net = torch.nn.DataParallel(net, device_ids=[5, 7])
    print('==> Build model finished')

    train(model=net, trainLoader=trainLoader, validLoader=None, lr=lr, epoch=epoch, modelPath=modelPath, valid=False, checkPoint=checkPoint)

