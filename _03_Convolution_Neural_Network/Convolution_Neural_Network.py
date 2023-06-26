# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/2 下午3:25

import torch.nn as nn


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


# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/4 下午12:59


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
# 定义使用GPU
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
epochs = 50
batch_size = 256
lr = 0.01

transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', transform=transform, download=True)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
# print(train_dataset[0])
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True, )
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
net = NeuralNetwork().cuda(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

train_loss = []
for epoch in range(epochs):
    sum_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        pred = net(x)

        optimizer.zero_grad()
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        train_loss.append(loss.item())
        print(["epoch:%d , batch:%d , loss:%.3f" % (epoch, batch_idx, loss.item())])
    torch.save(net.state_dict(), r'C:\Users\11620\convolution-neural-network-doushaaa\pth\model.pth')


def main():
    model = NeuralNetwork()  # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
