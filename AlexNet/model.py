import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):       #初始化函数来定义网络在正向传播过程中使用的层结构
        super(AlexNet, self).__init__()
        # n.Sequential能够将一系列的层结构进行打包组合成新的结构，如果像LeNet那样定义每一层会非常麻烦
        self.features = nn.Sequential(                              #这里对应特征提取
            #对照Alexnet所有层参数表进行设配置
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224] output[48, 55, 55]  由于花分类数据集较小，将卷积核个数减半，为了方便直接将padding设置为2
            nn.ReLU(inplace=True),                                  #inplace可以理解为pytorch增加计算量但降低内存使用
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]   这里卷积核个数同样减半，下同
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(           #将全连接层打包，组合成分类器
            nn.Dropout(p=0.5),                     #一般加载全连接层之间，默认失活比例为0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:                           #初始化权重
            self._initialize_weights()

    def forward(self, x):                          #正向传播过程
        x = self.features(x)                       #首先将输入的训练样本x输入到features中
        x = torch.flatten(x, start_dim=1)          #将上一步的输出进行展平处理，维度从dim=1开始(0维度为batch，1维度为channel)
        x = self.classifier(x)                     #展平后将其输入到分类结构中
        return x

    # 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():                  #遍历self.modules模块，通过self.modules模块会迭代定义的每一个层结构，判断其属于哪个类别
            if isinstance(m, nn.Conv2d):          #如果是卷积层，就用kaiming_normal对卷积权重w进行初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:            #如果偏置不为空，就用0对其进行初始化
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):        #如果是全连接层，那么采用normal对权重进行赋值，对偏置初始化为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
