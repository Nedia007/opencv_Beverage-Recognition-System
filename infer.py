# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import cv2
import numpy as np
import shutil
import glob
from tkinter import messagebox


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256))
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

        self.male_list = os.listdir(os.path.join(self.file_path, 'male'))
        self.female_list = os.listdir(os.path.join(self.file_path, 'female'))

        if btrain:
            self.male_list = self.male_list[0:3000]  # 取前3000作为训练集
            self.female_list = self.male_list[0:3000]
        else:
            self.male_list = self.male_list[3000:]  # 取后面的作为测试集
            self.female_list = self.female_list[3000:]

        self.img_list = []
        self.labels = []
        for male_img in self.male_list:
            male_img_path = os.path.join(os.path.join(self.file_path, 'male'), male_img)
            self.img_list.append(male_img_path)
            self.labels.append(0)
        for female_img in self.female_list:
            female_img_path = os.path.join(os.path.join(self.file_path, 'female'), female_img)
            self.img_list.append(female_img_path)
            self.labels.append(1)

        self.len = len(self.labels)
        self.transform = transforms.Compose([
            # 图像翻转与裁剪是图像增强的方法
            # transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.CenterCrop(224),  # 进行中心裁剪
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])

    def __getitem__(self, index):
        img_path = self.img_list[index]
        # print(img_path)
        img = read_image(img_path)
        img = self.transform(img)
        return img, self.labels[index], img_path

    def __len__(self):
        return self.len


def main():
    if torch.cuda.is_available():
        bgpu = True
        print('gpu is avaiable')
        print("友情提示 模型计算结果 0:代表male 1：代表female")

    #    classes = ('male', 'female')
    #    trainset =HumanDataset('E:\\PrivateCodesLocal\\ZSCTeaching_DL\\HumanDataset',True)
    #    testset = HumanDataset('E:\\PrivateCodesLocal\\ZSCTeaching_DL\\HumanDataset', False)

    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512, 2)
    
    net.load_state_dict(torch.load('net_  7d.pth'))
    net.eval()

    #    if bgpu:
    #        net = net.cuda()

    trans = transforms.Compose([
        # 图像翻转与裁剪是图像增强的方法
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.CenterCrop(224),  # 进行中心裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    cap = cv2.VideoCapture("2222.mp4")
    # VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频，如cap = cv2.VideoCapture(“../test.avi”)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 尺寸
    videoWriter = cv2.VideoWriter('hahaha_out2.mp4', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps,size)
    print('视频帧率为{},视频帧数为{},视频尺寸为{}'.format(fps, l, size))
    while cap.isOpened():
        ret, frame_org = cap.read()
        frame = cv2.resize(frame_org, size)
        input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input = trans(input)
        input = torch.unsqueeze(input, dim=0)
        #        if bgpu:
        #            input = input.cuda()

        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted.item())
        if predicted.item() == 0:
            cv2.putText(frame_org, 'cul:male', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 200, 200), 3)
            cv2.putText(frame_org, 'linyina', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 0), 3)
            cv2.putText(frame_org, '2018030402019', (150, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 100, 0), 3)
            cv2.putText(frame_org, 'real:female', (100, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 0, 0), 3)
        else:
            cv2.putText(frame_org, 'cul:female', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 200, 200), 3)
            cv2.putText(frame_org, 'linyina', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 0), 3)
            cv2.putText(frame_org, '2018030402019', (150, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 100, 0), 3)
            cv2.putText(frame_org, 'real:female', (100, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 0), 3)
        videoWriter.write(frame_org)
        video1 = cv2.imshow('output', frame_org)
        key = cv2.waitKey(50)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 测试
if __name__ == "__main__":
    main()
