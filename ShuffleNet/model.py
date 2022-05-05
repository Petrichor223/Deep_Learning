from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:               #channel_shuffle的实现

    batch_size, num_channels, height, width = x.size()               #获取传入的特征矩阵的size
    channels_per_group = num_channels // groups                      #将channel划分为groups组，channels_per_group对应每个组中channel的个数

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)   #改变通道排列顺序

    x = torch.transpose(x, 1, 2).contiguous()                           #通过transpose将维度1(groups)和维度2(channels_per_group)的信息调换，相当于转置

    # flatten
    x = x.view(batch_size, -1, height, width)                           #通过view方法将其还原成batch_size、channel、height、width

    return x


class InvertedResidual(nn.Module):                                      #shufflenet中的block
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:                                        #倒残差结构中步长只可能等1和2
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0                                        #判断输出特征矩阵channel是否是2的整数倍
        branch_features = output_c // 2                                 #因为concat左右两分支channel相同，如果相同则channel一定是2的整数倍
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)  #当s=2或input_channel是branch_features的两倍

        if self.stride == 2:                                            #当s=2，对应图d的block
            self.branch1 = nn.Sequential(                               #图d中左边分支
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),     #左边分支中的dw卷积
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  #左边分支中的1x1卷积
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()                               #当s=1，对应图c的左边分支

        self.branch2 = nn.Sequential(                                    #无论s=1或2，右边分支结构一样，但步长不一样
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,   #右边分支的第一个1x1卷积
                      stride=1, padding=0, bias=False),                  #当s=2时输入特征矩阵channel为input_c；当s=1时channel为branch_features
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),   #右边分支中的dw卷积
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),     #右边分支的第二个1x1卷积
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,              #实现图d中的dw卷积
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:                               #定义正向传播
        if self.stride == 1:                                              #s=1时，将输入特征矩阵channel进行均分处理
            x1, x2 = x.chunk(2, dim=1)                                    #通过chunk()进行均分成x1和x2，2为2等分，dim=1为通道排序中的维度
            out = torch.cat((x1, self.branch2(x2)), dim=1)                #对x1不做处理，将x2传入branch2得到其输出，然后在channel维度将其进行concat拼接
        else:                                                             #s=2时
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)                                     #将concat拼接得到的输出进行channel_shuffle处理

        return out


class ShuffleNetV2(nn.Module):                         #shufflenetv2网络搭建
    def __init__(self,
                 stages_repeats: List[int],            #stage2~4中block重复的次数
                 stages_out_channels: List[int],       #conv1、stage2~4、conv5对应的输出特征矩阵channel
                 num_classes: int = 1000,              #类别个数
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):     #搭建的block
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:                   #判断stages_repeats是否有三个参数(stage2~4)
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:              #判断stages_out_channels是否有五个参数(conv1、stage2~4、conv5)
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]  #conv1的输出特征矩阵channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels          #将当前的输出channel赋值给下一层的输入channel

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy     声明三个stage都是通过nn.Sequential实现的
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]                       #构建stage_name
        for name, repeats, output_channels in zip(stage_names, stages_repeats,       #通过for循环来搭建三个stage中的block
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]            #每个stage中的第一个block步长都是2，在Sequential中添加步长为2的block
            for i in range(repeats - 1):                                             #遍历剩下的block(步长都是1)
                seq.append(inverted_residual(output_channels, output_channels, 1))   #在Sequential中添加步长为1的block
            setattr(self, name, nn.Sequential(*seq))                                 #通过setattr给self设置一个变量，名字为name，值为nn.Sequential(*seq)
            input_channels = output_channels                                         #将当前的输出channel赋值给下一层的输入channel

        output_channels = self._stage_out_channels[-1]                               #将conv5输出特征矩阵channel赋值给当前的输出channel
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)          #全连接层

    def _forward_impl(self, x: Tensor) -> Tensor:                  #定义正向传播过程
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model
