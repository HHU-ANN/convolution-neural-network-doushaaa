import os
os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
from torchvision import datasets

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch


# 定义Alexnet网路结构
class NeuralNetwork(nn.Module):
    def __init__(self, width_mult=1):
        super(NeuralNetwork, self).__init__()
        # 定义每一个就卷积层
        self.layer1 = nn.Sequential(
            # 卷积层  #输入图像为1*28*28
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层特征图通道数不改变，每个特征图的分辨率变小
            # 激活函数Relu
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        # 定义全连接层
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        # 对应十个类别的输出

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# 设置超参数
epochs = 5
batch_size = 256
lr = 0.01

transform = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])


dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False,
                                           transform=torchvision.transforms.ToTensor())
data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = NeuralNetwork().cuda(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)

train_loss = []
for epoch in range(epochs):
    sum_loss = 0
    for batch_idx,(x,y) in enumerate(data_loader_train):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        optimizer.zero_grad()
        loss = loss_func(pred,y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        train_loss.append(loss.item())
        print(["epoch:%d , batch:%d , loss:%.3f" %(epoch,batch_idx,loss.item())])

torch.save(model.state_dict(), r'C:\Users\11620\convolution-neural-network-doushaaa\pth\model.pth')


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = NeuralNetwork()  # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model

