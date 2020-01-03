import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os
import csv
import torch.nn.functional as F

resultsave = './submission.csv'
modelfrom1 = './mymodel/first1222-6.pkl'
modelfrom2 = './mymodel/secondbest1224-3.pkl'
modelfrom3 = './mymodel/third1224-3.pkl'
testfile = './test'

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1*100*100*100
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),  # 10*32*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),  # 10*16*16*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3,padding=1),  # 50*20*20*20
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),  # 50*10*10*10
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,padding=1),  # 50*20*20*20
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),  # 50*3*3*3
        )
        self.fc1 = torch.nn.Linear(64 * 4 * 4 * 4, 64*4*2)
        # dropout消去某些连接，比例为p
        self.fc1_drop = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(64*4*2, 64)
        self.fc3 = torch.nn.Linear(64, 8)
        self.fc4 = torch.nn.Linear(8, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv2 = self.conv3(out_conv2)
        in_fc = out_conv2.view(out_conv2.size(0), -1)
        out = self.relu(self.fc1(in_fc))
        out_drop = self.fc1_drop(out)
        out = self.relu(self.fc2(out_drop))
        out_drop = self.fc1_drop(out)
        out =self.relu( self.fc3(out_drop))
        out_drop = self.fc1_drop(out)
        out = self.fc4(out_drop)
        return out

class secondNet(torch.nn.Module):
    def __init__(self):
        super(secondNet, self).__init__()
        # 1*32*32*32
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        #32*16*16*16
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        #128*8*8*8
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=256, out_channels=256, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        #256*2*2*2
        self.fc1 = torch.nn.Linear(256*2*2*2, 256)
        # dropout消去某些连接，比例为p
        #self.fc1_drop = torch.nn.Dropout(p=0)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.drop = torch.nn.Dropout3d(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        #out = self.fc1_drop(out)
        out = self.relu(self.fc2(out))
        out = self.drop(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

    # conv&fc need relu while drop&pool don't

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    # 32, 64, 1; 32*16*16*16
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #64*16*16*16
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #64*16*16*16
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 32
        super(ResNet, self).__init__()
        # 1*32*32*32
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        #32*16*16*16
        self.layer1 = self._make_layer(block, 64, layers[0])
        #64*16*16*16
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sig = nn.Sigmoid()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.sig(x)

        return x

def resnet0():
    return ResNet(BasicBlock, [2,2,2,2])

def testmodel(modelfrom1, modelfrom2, modelfrom3):

    model1 = torch.load(modelfrom1)
    model2 = torch.load(modelfrom2)
    model3 = torch.load(modelfrom3)
    with open(resultsave, "w", newline='') as f:
        writer = csv.writer(f)
        header = ['Id', 'Predicted']
        writer.writerows([header])
        for file in os.listdir(testfile):
            # id
            datahead = []
            qian2 = file.split('.')[0]
            # hou = qian2.split('e')[-1]
            datahead.append(qian2)
            # calculate
            tmp = np.load(os.path.join(testfile, file))
            img = tmp['voxel']
            img2 = tmp['seg']
            img = img * img2
            img3 = img[34:66, 34:66, 34:66]
            img3 = transforms.ToTensor()(img3)  # 100*100*100
            # img = transforms.Normalize(mean=[.5], std=[.5])(img)
            img3 = img3.unsqueeze(0)
            img3 = img3.unsqueeze(0)
            model1.eval()
            model2.eval()
            model3.eval()

            output = model1(img3)
            '''output现在可以是负数'''
            output = F.log_softmax(output, dim=1)
            pre1 = output[0][0] / (output[0][0] + output[0][1])
            pre1 = float(pre1)

            output = model2(img3)
            '''output现在可以是负数'''
            output = F.log_softmax(output, dim=1)
            pre2 = output[0][0] / (output[0][0] + output[0][1])
            pre2 = float(pre2)

            output = model3(img3)
            '''output现在可以是负数'''
            output = F.log_softmax(output, dim=1)
            pre3 = output[0][0] / (output[0][0] + output[0][1])
            pre3 = float(pre3)

            pre = (pre1+pre2+pre3)/3

            datahead.append(pre)
            # datahead.append(hou)
            writer.writerows([datahead])
        f.close()

testmodel(modelfrom1, modelfrom2, modelfrom3)
