import torch as th
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


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
        transforms.
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

def getTrainValidDataLoader(data_dir, batch_size, inputsize, augment, random_seed=123,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # define transforms
    valid_transform = transforms.Compose([transforms.Resize(inputsize), transforms.ToTensor(), normalize])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(inputsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(inputsize),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=data_dir, transform=valid_transform)
    classNum = len(train_dataset.classes)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = th.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = th.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, classNum


def loadWebface(rootPath, batchSize, inputsize):

    data_transform = transforms.Compose([
        transforms.Resize(inputsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=rootPath, transform=data_transform)
    classNum = len(dataset.classes)
    datasetLoader = th.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)

    return datasetLoader, classNum


if __name__ == '__main__':

    rootPath = "data/CASIA-WebFace/"
    batchSize = 8

    dataLoader = loadWebface(rootPath, batchSize, inputsize=(112, 96))
    for inputs, targets in dataLoader:
        print("input", inputs.size())
        print("target", targets)

