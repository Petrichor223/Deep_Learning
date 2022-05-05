import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

# http://download.tensorflow.org/example_images/flower_photos.tgz
root = "D:/Dataset/flower_photos"  # 数据集所在根目录


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")         #判断当前设备是否有GPU，有就用没有就不用
    print("using {} device.".format(device))                                      #打印使用的设备信息

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)  #传入数据集，通过read_split_data方法划分训练集和验证集

    data_transform = {                                                             #定义训练集和验证集预处理方法
        "train": transforms.Compose([transforms.RandomResizedCrop(224),            #随机裁剪到224x224
                                     transforms.RandomHorizontalFlip(),            #随机水平翻转
                                     transforms.ToTensor(),                        #转化成Tensor格式
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),   #标准化处理
        "val": transforms.Compose([transforms.Resize(256),                         #将最小边resize到256x256
                                   transforms.CenterCrop(224),                     #进行中心裁剪到224x224
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,                #训练集图像的路径列表
                               images_class=train_images_label,              #训练集图像的标签信息
                               transform=data_transform["train"])            #预处理方法

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,            #shuffle=True在训练过程中会打乱数据集的顺序
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for step, data in enumerate(train_loader):
        images, labels = data


if __name__ == '__main__':
    main()
