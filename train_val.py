'''
wenti:在benchsize不同时，loss会不一样
'''
from builtins import int

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

if_test = True
newmodel = False
modelfrom = './first1222-6.pkl'
modelsave = './second5.pkl'
resultsave = 'first1222-6.csv'
learningrate = 0.01
bestrate = 0.63
bestloss = 0.095
bestsave = './bestmodel.pkl'
ifbest = False
NUM_EPOCHS = 100  # 总样本循环次数
batch_size = 16  # 训练时的一组数据的大小


# todo：搭建神经网络
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

        return x

def resnet0():
    return ResNet(BasicBlock, [2,2,2,2])
model = torch.load(modelfrom)

# todo：定义优化器和损失
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
loss_fun = torch.nn.CrossEntropyLoss()
'''重要，交叉熵函数已经自带了Softmax–Log–NLLLoss的过程'''
# criterion = torch.nn.NLLLoss()
# optimizer = torch.optim.Adam(model.parameters())  # 优化方法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# todo：重载数据
class mydataset(Dataset):
    def __init__(self, image_dir):
        self.train = image_dir
        self.train_name = []
        self.train_id = []
        self.csv_name = {}
        # train_name 存储了样本的带npz名字，用于读取数据；train_id存储了样本的名字，用于查找标签
        for file in os.listdir(self.train):
            qian = file.split('.')[0]
            self.train_name.append(file)
            self.train_id.append(qian)
        # csv_name是字典
        with open("train_val.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.csv_name[line[0]] = int(line[1])
            csvfile.close()
        self.lenth = len(self.train_id)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[.5], std=[.5])

    def __getitem__(self, index):
        file_name = self.train_name[index]
        file_id = self.train_id[index]
        tmp = np.load(os.path.join(self.train, file_name))
        img = tmp['voxel']
        img2 = tmp['seg']
        img = img * img2
        img3 = img[34:66, 34:66, 34:66]
        img3 = self.data_preproccess(img3)

        # img = transforms.Normalize(mean=[.5], std=[.5])(img)
        # label = []
        # label.append(self.csv_name[file_id])
        # label = np.array(self.csv_name[file_id])
        # label = self.data_preproccess(label)
        label = self.csv_name[file_id]
        return img3, label

    def __len__(self):
        return self.lenth

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data

# todo：载入数据
time1 = time.time()
train_data = mydataset('./train_val')
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = mydataset('./test_val')
test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def testtrain():
    print('trainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
    model.eval()
    test_loss = 0
    correct = 0
    for images, labels in tqdm(train_loader):
        images = images.unsqueeze(1)
        output = model(images)

        # 计算test——loss和结果正确的数 模板
        test_loss += loss_fun(output, labels).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    print('average_loss = %0.5f%%' % (test_loss / len(train_loader.dataset) * 100))
    print('accuracy = %0.5f%%' % (correct / len(train_loader.dataset) * 100))

def testval():
    model.eval()
    test_loss = 0
    correct = 0
    for images, labels in test_loader:
        images = images.unsqueeze(1)
        output = model(images)

        # 计算test——loss和结果正确的数 模板
        test_loss += loss_fun(output, labels).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    print('average_loss = %0.5f%%' % (test_loss / len(test_loader.dataset) * 100))
    print('accuracy = %0.5f%%' % (correct / len(test_loader.dataset) * 100))

# todo:test
testtrain()
testval()

# todo:getresult
# for file in os.listdir('./test'):
if if_test:
    with open(resultsave, "w", newline='') as f:
        writer = csv.writer(f)
        header = ['Id', 'Predicted', 'num']
        writer.writerows([header])
        for file in os.listdir('./test'):
            #id
            datahead = []
            qian2 = file.split('.')[0]
            hou = qian2.split('e')[-1]
            datahead.append(qian2)
            #calculate
            tmp = np.load(os.path.join('./test', file))
            img = tmp['voxel']
            img2 = tmp['seg']
            img = img * img2
            img3 = img[34:66, 34:66, 34:66]
            img3 = transforms.ToTensor()(img3) #100*100*100
            #img = transforms.Normalize(mean=[.5], std=[.5])(img)
            img3 = img3.unsqueeze(0)
            img3 = img3.unsqueeze(0)
            model.eval()

            output = model(img3)
            '''output现在可以是负数'''
            output = F.log_softmax(output, dim=1)

            pre = output[0][0]/(output[0][0]+output[0][1])
            pre = float(pre)
            datahead.append(pre)
            datahead.append(hou)
            writer.writerows([datahead])
        f.close()

time2 = time.time()
print('time_used = ', (time2 - time1))
print(if_test)
"""
1.要查看model的来源是哪一个
2.要查看model的两次保存
3.要检查bench——size和ecoph
4.要检查输出文件
"""