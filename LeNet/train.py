import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import time


#def main():
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000张训练图片
# 第一次使用时要将download设置为True才会自动去下载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                        shuffle=True, num_workers=0)

# 10000张验证图片
# 第一次使用时要将download设置为True才会自动去下载数据集
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                        shuffle=False, num_workers=0)
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = LeNet()                                                     #实例化模型
net.to(device)                                                    #将网络分配到指定的device中
loss_function = nn.CrossEntropyLoss()                             #定义损失函数，在nn.CrossEntropyLoss中已经包含了Softmax函数
optimizer = optim.Adam(net.parameters(), lr=0.001)                #定义优化器，这里使用Adam优化器，net是定义的LeNet，parameters将LeNet所有可训练的参数都进行训练，lr=learning rate

for epoch in range(5):  # loop over the dataset multiple times    #将训练集迭代的次数

    running_loss = 0.0                                            #累加训练过程的损失
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):           #遍历训练集样本
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data                                     #将得到的数据分离成输入和标签

        # zero the parameter gradients
        optimizer.zero_grad()                                     #将历史损失梯度清零，如果不清楚历史梯度，就会对计算的历史梯度进行累加(通过这个特性能够变相实现一个很大的batch)
        # forward + backward + optimize
        outputs = net(inputs.to(device))                          # 将inputs分配到指定的device中
        loss = loss_function(outputs, labels.to(device))          # 将labels分配到指定的device中
        #outputs = net(inputs)                                    #将图片输入到网络进行正向传播得到输出
        #loss = loss_function(outputs, labels)                    #计算损失，outputs为网络预测值，labels为输入图片对应的真实标签
        loss.backward()                                           #将loss进行反向传播
        optimizer.step()                                          #进行参数更新

        # print statistics
        running_loss += loss.item()                               #计算完loss完之后将其累加到running_loss
        if step % 500 == 499:    # print every 500 mini-batches   #每隔500次打印一次训练的信息
            with torch.no_grad():                                 #with是一个上下文管理器
                outputs = net(val_image.to(device))               # 将test_image分配到指定的device中
                #outputs = net(val_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]          #在维度1上进行最大值的预测，[1]为index索引
                accuracy = (predict_y == val_label.to(device)).sum().item() / val_label.size(0)  # 将test_label分配到指定的device中
                #accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)      #将预测的标签类别与真实的标签类别进行比较，在相同的地方返回值为1，否则为0，用此计算预测对了多少样本

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))

                print('%f s' % (time.perf_counter() - time_start))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'                                        #保存权重
torch.save(net.state_dict(), save_path)                          #将网络的所有参数及逆行保存


#
# if __name__ == '__main__':
#     main()