import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):                                                      #定义GoogLeNet
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):   #初始化函数
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits                                             #将是否使用辅助分类器的布尔变量传入类变量中
                                                                                 #根据GoogLeNet简图进行搭建
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)      #为了将特征矩阵的高和宽缩减到原来的一半，这里将padding设置为3：(224-7+2*3)/2+1=112.5(pytorch默认向下取正)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)                #ceil_mode表示如果进行最大池化后得到的值为小数，设置为True就会向上取整，设置为False就会向下取整
                                                                                 #省略LocalRespNorm，没什么用
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)               #使用定义的Inception模板
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)             #这里每一个Inception的输入都可以通过将上一层Inception层的四个分支的特征矩阵深度加起来得到
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:                                                    #如果使用辅助分类器，即aux_logits = True，则创建aux1和aux2
            self.aux1 = InceptionAux(512, num_classes)                         #输入是Inception4a的输出
            self.aux2 = InceptionAux(528, num_classes)                         #输入是Inception4b的输出

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                            #nn.AdaptiveAvgPool2d()自适应平均池化下采样，参数(1,1)代表我们所需要的输出特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)                                         #不论输入特征矩阵的高和宽的大小，都可以通过自适应平均池化下采样得到所指定的输出特征矩阵的高和宽
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:                                                       #如果init_weights = True，则对模型权重进行初始化
            self._initialize_weights()

    def forward(self, x):                                                      #定义正向传播过程
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer   是否使用辅助分类器，在训练过程使用，测试过程不用
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer    是否使用辅助分类器，在训练过程使用，测试过程不用
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):                 #定义Inception结构模板
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):  #结合googlenet网络参数和Inception结构
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)    #分支1，使用定义的卷积模板，输入的特征矩阵深度为in_channels，卷积核个数为传入的ch1x1

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),           #分支2
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)       #将padding设置为1，使输出特征矩阵和输入特征矩阵的高和宽保持一致，保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(                                    #分支3
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)       #保证输出大小等于输入大小：output_size=(input_size-5+2*2)/1+1=input_size
        )

        self.branch4 = nn.Sequential(                                    #分支4
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):                         #定义正向传播过程
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):                                    #定义辅助分类器
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)  #平均池化
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)                          #输入节点个数128x4x4
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):                                         #定义正向传播过程
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)            #原论文采用0.7
        # N x 2048                                               #self.training会随着训练或测试的不同而变化
        x = F.relu(self.fc1(x), inplace=True)                    #当实例化一个模型model后，可以通过model.train()和model.eval()来控制模型的状态
        x = F.dropout(x, 0.5, training=self.training)            #在model.train()模式下self.training=True，在model.eval()模式下self.training=False
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):        #在搭建网络之前先进行模板文件的创建，在搭建卷积层过程中通常将卷积和ReLU激活函数共同使用，可以通过此方法定义卷积模板
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):            #定义正向传播过程
        x = self.conv(x)
        x = self.relu(x)
        return x
