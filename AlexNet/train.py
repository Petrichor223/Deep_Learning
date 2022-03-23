import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     #有GPU就用，没有就不用
    print("using {} device.".format(device))
    #对数据进行预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),      #随机裁剪成224x224大小
                                     transforms.RandomHorizontalFlip(),      #随机翻转，水平方向
                                     transforms.ToTensor(),                  #转化成Tensor
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}    #标准化处理
    #os.getcwd()获取当前文件所在的目录；os.path.join()将输入的路径连接在一起；..表示返回上层目录；../..表示返回上上层目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path获取数据集所在的根目录
    image_path = os.path.join(data_root, "./flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),   #加载数据集路径
                                         transform=data_transform["train"])        #数据预处理
    train_num = len(train_dataset)                                                 #通过len()函数打印训练集有多少张图片

    # # 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx                               # 将 flower_list 中的 key 和 val 调换位置
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)                              # 将 cla_dict 写入 json 文件中，方便在预测时读取信息
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    #载入train_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    #载入测试集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    #查看数据集，查看之前要把batch_size修改成4，随机打乱shuffle改成True
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)    #实例化网络，数据集有五个类别，初始化权重

    net.to(device)                                     #将网络指定到设备上
    loss_function = nn.CrossEntropyLoss()              #定义损失函数，
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)   #Adam优化器

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0                                      #最佳准确率，在训练网络过程中保存准确率最高的模型
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train  在网络搭建中使用了dropout方法，会在正向传播过程中随机失活一部分神经。但是只希望在训练过程中使用，在预测过程中不使用，因此通过net.train和net.eval来管理dropout(也可以管理BN层)
        net.train()                                              #在训练过程中调用net.train会启用dropout
        running_loss = 0.0                                       #统计在训练过程中的平均损失
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):                  #遍历数据集
            images, labels = data                                #将数据集分为图像和标签
            optimizer.zero_grad()                                #清空之前的梯度信息进行正向传播
            outputs = net(images.to(device))                     #将训练图像指定到设备上
            loss = loss_function(outputs, labels.to(device))     #计算预测值与真实值之间的损失
            loss.backward()                                      #将得到的损失反向传播到每个节点中
            optimizer.step()                                     #更新每个节点的参数

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,    #打印训练过程中的训练进度
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()                                               #在验证过程中调用net.eval会关闭dropout
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():                                    #禁止pytorch对参数跟踪，即在验证过程中不进行损失计算
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:                             #遍历验证集对图片进行划分
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]         #求得输出的最大值作为预测
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  #将预测与真是标签作对比

        val_accurate = acc / val_num                             #求测试集准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:                             #求得最大准确率，保存其权重
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
