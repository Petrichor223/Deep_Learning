import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):         #在Pytorch中搭建模型首先要定义一个类，这个类要继承于nn.Module这个副类
    def __init__(self):         #在该类中首先要初始化函数，实现在搭建网络过程中需要使用到的网络层结构，#然后在forward中定义正向传播的过程
        super(LeNet, self).__init__()      #super能够解决在多重继承中调用副类可能出现的问题
        self.conv1 = nn.Conv2d(3, 16, 5)   #这里输入深度为3，卷积核个数为16，大小为5x5
        self.pool1 = nn.MaxPool2d(2, 2)    #最大池化核大小为2x2，步长为2
        self.conv2 = nn.Conv2d(16, 32, 5)  #经过Conv2d的16个卷积核处理后，输入深度变为16
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  #全连接层的输入是一维的向量，因此将输入的特征矩阵进行展平处理(32x5x5)，然后根据网络设置输出
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)       #输出有几个类别就设置几

    def forward(self, x):     #在forward中定义正向传播的过程
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)    可通过矩阵尺寸大小计算公式得
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


#测试
#input1 =torch.rand([32,3,32,32])     #定义随机生成数据的shape
#model = LeNet()                      #实例化模型
#print(model)
#output = model(input1)               #将数据输入到网络中进行正向传播
