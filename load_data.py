import torch as th
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms


def loadMNIST(batchSize):

    root = "./data/"

    trans= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=trans)

    train_loader = th.utils.data.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True)
    test_loader = th.utils.data.DataLoader(dataset=test_set, batch_size=batchSize, shuffle=False)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total testing batch number: {}'.format(len(test_loader)))

    return train_loader, test_loader


def loadCIFAR10(batchSize):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def getMean_Std(rootPath):

    data_transform = transforms.Compose([transforms.RandomSizedCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=rootPath, transform=data_transform)
    print(dataset.classes)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = th.zeros(3)
    std = th.zeros(3)
    print('==> Computing mean and std..')

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std

def loadWebface(rootPath, batchSize, inputsize):

    data_transform = transforms.Compose([
        transforms.Resize((inputsize, inputsize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=rootPath, transform=data_transform)
    classNum = len(dataset.classes)
    datasetLoader = th.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)

    return datasetLoader, classNum

if __name__ == '__main__':

    rootPath = "data/CASIA-WebFace/"
    # atchSize = 8

    mean, std = getMean_Std(rootPath)
    print(mean)
    print(std)

    # dataLoader = loadWebface(rootPath, batchSize)
    # for inputs, targets in dataLoader:
    #     print("input", inputs.size())
    #     print("target", targets)

