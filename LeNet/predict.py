import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),               #首先需resize成跟训练集图像一样的大小
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()                                    #实例化网络
    net.load_state_dict(torch.load('Lenet.pth'))     #载入保存的权重文件

    im = Image.open('1.jpg')                         #导入要测试的图片
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]  #对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
