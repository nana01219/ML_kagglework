#
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

# 今天的改进方向1）改正一下lr吗2）修改weightdecay吗

newmodel = True
modelfrom = './second1224-1pkl'
modelsave = './second1224-3.pkl'
resultsave = './second1224-3.csv'
learningrate = 1e-4
bestrate = 0.60
bestloss = 0.10
trainrate = 0.75
trainloss = 0.05
bestsave = './secondbest1224-3.pkl'
ifbest = True # 是否earlystop
NUM_EPOCHS = 50  # 总样本循环次数
batch_size = 8  # 训练时的一组数据的大小
use_mixup = True


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

if newmodel:
    model = secondNet()
else:
    model = torch.load(modelfrom)

# todo：定义优化器和损失
#, weight_decay=0.025
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
loss_fun = torch.nn.CrossEntropyLoss()
'''重要，交叉熵函数已经自带了Softmax–Log–NLLLoss的过程'''
# criterion = torch.nn.NLLLoss()
# optimizer = torch.optim.Adam(model.parameters())  # 优化方法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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

def mixup_data(x, y, alpha=0.5, use_cuda=False):
    # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]    # 此处是对数据x_i 进行操作
    y_a, y_b = y, y[index]    # 记录下y_i 和y_j
    return mixed_x, y_a, y_b, lam    # 返回y_i 和y_j 以及lambda

# todo：载入数据
time1 = time.time()
train_data = mydataset('./train_val')
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = mydataset('./test_val')
test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# full_dataset = mydataset('./train_val')
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
# train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def testtrain():
    print('trainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
    model.eval()
    test_loss = 0
    correct = 0
    for images, labels in train_loader:
        images = images.unsqueeze(1)
        output = model(images)

        # 计算test——loss和结果正确的数 模板
        test_loss += loss_fun(output, labels).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    print('average_loss = %0.5f%%' % (test_loss / len(train_loader.dataset) * 100))
    print('accuracy = %0.5f%%' % (correct / len(train_loader.dataset) * 100))

    acc1 = correct / len(train_loader.dataset)
    los1 = test_loss / len(train_loader.dataset)

    global bestrate
    global bestloss
    global bestsave
    global ifbest
    global trainrate

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

    # find the best
    if ifbest:
        acc = correct / len(test_loader.dataset)
        los = test_loss / len(test_loader.dataset)
        if (acc >= bestrate) & (los <= bestloss) & (acc1 >= trainrate):
            bestrate = acc
            #bestloss = los
            torch.save(model, bestsave)


    print('average_loss = %0.5f%%' % (test_loss / len(test_loader.dataset) * 100))
    print('accuracy = %0.5f%%' % (correct / len(test_loader.dataset) * 100))

# todo：train
for epoch in range(NUM_EPOCHS):
    print(epoch)
    model.train()

    for images, labels in train_loader:
        images = images.unsqueeze(1)
        if use_mixup:
            images, label_a, label_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()
            output = model(images)
            loss = mixup_criterion(loss_fun, output, label_a, label_b, lam)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()  # 所有参数的梯度清零
            output = model(images)
            loss = loss_fun(output, labels)
            loss.backward()  # 即反向传播求梯度
            optimizer.step()  # 调用optimizer进行梯度下降更新参数

    testtrain()

torch.save(model, modelsave)
testtrain()

# todo:getresult
# for file in os.listdir('./test'):
with open(resultsave, "w", newline='') as f:
    writer = csv.writer(f)
    header = ['Id', 'Predicted', 'num']
    writer.writerows([header])
    for file in os.listdir('./test'):
        # id
        datahead = []
        qian2 = file.split('.')[0]
        hou = qian2.split('e')[-1]
        datahead.append(qian2)
        # calculate
        tmp = np.load(os.path.join('./test', file))
        img = tmp['voxel']
        img2 = tmp['seg']
        img = img * img2
        img3 = img[34:66, 34:66, 34:66]
        img3 = transforms.ToTensor()(img3)  # 100*100*100
        # img = transforms.Normalize(mean=[.5], std=[.5])(img)
        img3 = img3.unsqueeze(0)
        img3 = img3.unsqueeze(0)
        model.eval()

        output = model(img3)
        '''output现在可以是负数'''
        output = F.log_softmax(output, dim=1)

        pre = output[0][0] / (output[0][0] + output[0][1])
        pre = float(pre)
        datahead.append(pre)
        datahead.append(hou)
        writer.writerows([datahead])
    f.close()

time2 = time.time()
print('time_used = ', (time2 - time1))
print(ifbest)
"""
1.要查看model的来源是哪一个
2.要查看model的两次保存
3.要检查bench——size和ecoph
4.要检查输出文件
"""
