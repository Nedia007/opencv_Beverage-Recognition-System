# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


import torch.optim as optim


import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset
from PIL import Image
import cv2
import numpy as np
import shutil
import glob


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256,256))
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


'''
性别分类数据集
male 3698    train 3000    test 698
female 3810   trian  3000    test 810
'''
class HumanDataset(Dataset):
    def __init__(self, file_path, btrain):
        self.file_path = file_path
        self.btrain = btrain

        self.male_list = os.listdir(os.path.join(self.file_path,'male'))
        self.female_list = os.listdir(os.path.join(self.file_path, 'female'))

        if btrain:
            self.male_list = self.male_list[0:3000]  #取前3000作为训练集
            self.female_list = self.female_list[0:3000]
        else:
            self.male_list = self.male_list[3000:]   #取后面的作为测试集
            self.female_list = self.female_list[3000:]

        self.img_list = []
        self.labels = []
        for male_img in self.male_list:
            male_img_path = os.path.join(os.path.join(self.file_path,'male'),male_img)
            self.img_list.append(male_img_path)
            self.labels.append(0)
        for female_img in self.female_list:
            female_img_path = os.path.join(os.path.join(self.file_path,'female'),female_img)
            self.img_list.append(female_img_path)
            self.labels.append(1)

        self.len = len(self.labels)
        self.transform = transforms.Compose([
            #图像翻转与裁剪是图像增强的方法
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.CenterCrop(224),  #进行中心裁剪
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])

    def __getitem__(self,index):
        img_path = self.img_list[index]
        # print(img_path)
        img = read_image(img_path)
        img = self.transform(img)
        return img,self.labels[index],img_path

    def __len__(self):
        return  self.len


def main():
    # 超参数设置
    # EPOCH = args.Epoch  # 遍历数据集次数


    classes = ('male', 'female')
    trainset =HumanDataset('D:\\Users\\lina\\Desktop\\HumanDataset',True)
    testset = HumanDataset('D:\\Users\\lina\\Desktop\\HumanDataset', False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    if torch.cuda.is_available():
        bgpu = True
        print('gpu is avaiable')
    else:
        bgpu = False
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512, 2)  
    if bgpu:
        net = net.cuda()

    EPOCH = 10
    BATCH_SIZE = 16
    LR = 0.01  # 学习率
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    best_acc = 60  # 2 初始化best test accuracy
    print("Start Training, Resnet-18 to classify gender!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, (inputs, labels, inputs_path) in enumerate(trainloader):
                    # 准备数据
                    length = len(trainloader)

                    #显示每个batch的第一张图片，验证输入图片是否正确
                    # print(inputs.shape)
                    # T= torchvision.transforms.ToPILImage()
                    # for i_img in range(inputs.shape[0]):
                    #     input_img = T(inputs[0, :, :, :])
                    #     input_img = cv2.cvtColor(np.asarray(input_img), cv2.COLOR_RGB2BGR)
                    #     cv2.imshow('input',input_img)
                    #     cv2.waitKey(10)

                    if bgpu:
                        inputs,labels = inputs.cuda(),labels.cuda()

                    optimizer.zero_grad()
                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0

                    for data in testloader:
                        net.eval()
                        images, labels,images_path = data
                        if bgpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        print(labels, predicted)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：{}, [{}/{}]'.format (100 * correct / total,correct,total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), 'net_{:3d}d.pth' .format(epoch + 1))  #保存模型
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中

                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
# 训练
if __name__ == "__main__":
	main()
