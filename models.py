import torch
import torch.nn as nn
import torch.nn.functional as F

class AAA(nn.Module):
    def __init__(self):
        super(AAA, self).__init__()

        self.conv_1 = nn.Conv2d(1, 16, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

        self.conv_2 = nn.Conv2d(16, 32, 5)

        self.fc_1 = nn.Linear(16*32, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):

        x = self.relu(self.conv_1(x))
        x = self.pool(x)

        x = self.relu(self.conv_2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 16*32)

        x = self.fc_1(x)
        x = self.output(x)

        return x

    def ConvPart(self, x):
        x = self.relu(self.conv_1(x))
        x = self.pool(x)

        x = self.relu(self.conv_2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 16*32)

        x = self.fc_1(x)
        return x

    def FullyPart(self, x):
        return self.output(x)

class CLA(nn.Module):
    def __init__(self):
        super(CLA, self).__init__()

        self.conv2d_1 = nn.Conv2d(1, 16, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

        self.conv2d_2 = nn.Conv2d(16, 4, 5)

        self.dense_1 = nn.Linear(16*4, 512)
        self.dense_2 = nn.Linear(512, 10)

        self.drop = nn.Dropout(0.0)

    def forward(self, x):

        x = self.relu(self.conv2d_1(x))
        x = self.pool(x)

        x = self.relu(self.conv2d_2(x))
        x = self.pool(x)
        x = self.drop(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 16*4)

        x = self.relu(self.dense_1(x))
        x = self.drop(x)
        x = self.dense_2(x)

        return x

    def ConvPart(self, x):

        x = self.relu(self.conv2d_1(x))
        x = self.pool(x)

        x = self.relu(self.conv2d_2(x))
        x = self.pool(x)
        x = self.drop(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 16*4)

        x = self.relu(self.dense_1(x))
        x = self.drop(x)
        return x

    def FullyPart(self, x):
        x = self.dense_2(x)

        return x
        

class Kwon(nn.Module):
    def __init__(self):
        super(Kwon, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.relu = nn.ReLU(inplace = True)
        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 10)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(-1, 64*4*4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

    def ConvPart(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(-1, 64*4*4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def FullyPart(self, x):
        return self.out(x)

class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()

        self.conv2d_1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.conv2d_2 = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.conv2d_3 = nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0)
        self.conv2d_4 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        self.conv2d_5 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv2d_6 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv2d_7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv2d_8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv2d_9 = nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(0.5)
        self.avepool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.relu = nn.ReLU(inplace = True)


    def forward(self, x):
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.drop(self.maxpool(x))

        x = self.relu(self.conv2d_4(x))
        x = self.relu(self.conv2d_5(x))
        x = self.relu(self.conv2d_6(x))
        x = self.drop(self.maxpool(x))

        x = self.relu(self.conv2d_7(x))
        x = self.relu(self.conv2d_8(x))
        x = self.relu(self.conv2d_9(x))
        x = self.avepool(x)
        x = x.view(x.size(0), 10)

        return x

    def ConvPart(self, x):
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.drop(self.maxpool(x))

        x = self.relu(self.conv2d_4(x))
        x = self.relu(self.conv2d_5(x))
        x = self.relu(self.conv2d_6(x))
        x = self.drop(self.maxpool(x))

        x = self.relu(self.conv2d_7(x))
        x = self.relu(self.conv2d_8(x))
        x = self.relu(self.conv2d_9(x))

        return x

    def FullyPart(self, x):
        x = self.avepool(x)
        x = x.view(x.size(0), 10)

        return x
   

class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, 3, stride=1, padding = 1)
        self.conv2d_2 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv2d_3 = nn.Conv2d(32, 64, 3, stride=1, padding = 1)
        self.conv2d_4 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv2d_5 = nn.Conv2d(64, 128, 3, stride=1, padding = 1)
        self.conv2d_6 = nn.Conv2d(128, 128, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense_1 = nn.Linear(512, 512)
        self.dense_2 = nn.Linear(512, 43)

    def forward(self, x):

        x = F.relu(self.conv2d_1(x))
        x = self.pool(F.relu(self.conv2d_2(x)))
        x = F.relu(self.conv2d_3(x))
        x = self.pool(F.relu(self.conv2d_4(x)))
        x = F.relu(self.conv2d_5(x))
        x = self.pool(F.relu(self.conv2d_6(x)))

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 512)

        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)

        return x

    def ConvPart(self, x):

        x = F.relu(self.conv2d_1(x))
        x = self.pool(F.relu(self.conv2d_2(x)))
        x = F.relu(self.conv2d_3(x))
        x = self.pool(F.relu(self.conv2d_4(x)))
        x = F.relu(self.conv2d_5(x))
        x = self.pool(F.relu(self.conv2d_6(x)))

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 512)

        x = F.relu(self.dense_1(x))

        return x

    def FullyPart(self, x):
        x = self.dense_2(x)

        return x

class GTSRB(nn.Module):

    def __init__(self):
        super(GTSRB, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, stride=1)
        self.conv_2 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv_3 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv_4 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv_5 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv_6 = nn.Conv2d(128, 128, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_1 = nn.Linear(128, 512)
        self.output = nn.Linear(512, 43)

    def forward(self, x):

        x = F.relu(self.conv_1(x))
        x = self.pool(F.relu(self.conv_2(x)))
        x = F.relu(self.conv_3(x))
        x = self.pool(F.relu(self.conv_4(x)))
        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        x = F.relu(self.fc_1(x))
        x = self.output(x)

        return x

    def ConvPart(self, x):

        x = F.relu(self.conv_1(x))
        x = self.pool(F.relu(self.conv_2(x)))
        x = F.relu(self.conv_3(x))
        x = self.pool(F.relu(self.conv_4(x)))
        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        x = F.relu(self.fc_1(x))

        return x

    def FullyPart(self, x):
        x = self.output(x)

        return x


class DeepID(nn.Module):

    def __init__(self):
        super(DeepID, self).__init__()
        self.conv_1 = nn.Conv2d(3, 20, 4, stride=1)
        self.conv_2 = nn.Conv2d(20, 40, 3, stride=1)
        self.conv_3 = nn.Conv2d(40, 60, 3, stride=1)
        self.conv_4 = nn.Conv2d(60, 80, 2, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_1 = nn.Linear(1200, 160)
        self.fc_2 = nn.Linear(960, 160)
        self.output = nn.Linear(160, 1283)

    def forward(self, x):

        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        y = F.relu(self.conv_4(x))

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 1200)
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, 960)

        x = self.fc_1(x)
        y = self.fc_2(y)

        add = F.relu(torch.add(x, y))

        add = self.output(add)

        return add

    def ConvPart(self, x):

        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        y = F.relu(self.conv_4(x))

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 1200)
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, 960)

        x = self.fc_1(x)
        y = self.fc_2(y)

        add = F.relu(torch.add(x, y))

        return add

    def FullyPart(self, x):

        add = self.output(x)

        return add
