import torch
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cuda import *

try:
    os.mkdir('a2o')
except:
    pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

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

def poison(x, y, a2a):
    xp, yp = x.clone(), y.clone()
    xp[:, :,26,26] = 1
    xp[:, :, 25,25] = 1
    xp[:, :, 24,26] = 1
    xp[:, :, 26,24] = 1
    if a2a:
        yp += 1
        yp = yp % 10
    else:
        yp *= 0
    
    return xp, yp

def categorize(dataload):
    inputs, labels = [], []
    for (x,y) in dataload:
        inputs.append(x)
        labels.append(y)
    inputs = torch.cat(inputs, 0)
    labels = torch.cat(labels, 0)
    return (inputs, labels)


def share(party, pct = 0.1):
    x, y = party
    shared = []
    for i in range(10):
        indices = (y==i).nonzero()[:,0]
        l = int(len(indices)*pct)
        shared.append(indices[:l])
    shared = torch.cat(shared, 0)
    return x[shared], y[shared]

batch_size = 1000

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform = transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform = transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True)


x, y = categorize(trainloader)
valx, valy = x[:5000], y[:5000]
x, y = x[5000:], y[5000:]
xt, yt = categorize(testloader)

x, xt = x.reshape(-1, 1, 28, 28), xt.reshape(-1, 1, 28, 28)
torch.save(xt, 'data/clean_inputs.pt')
torch.save(yt, 'data/clean_labels.pt')
torch.save(valx, 'data/val_inputs.pt')
torch.save(valy, 'data/val_labels.pt')

tmpx, tmpy = share((x, y), pct = 0.15)

a2a = False
xp, yp = poison(tmpx, tmpy, a2a)
xtp, ytp = poison(xt, yt, a2a)

clean, poisons  = xt[0,0], xtp[0,0]
plt.figure()
plt.imshow(clean, cmap = 'gray', vmin =0, vmax=1)
plt.savefig('clean.png')
plt.figure()
plt.imshow(poisons, cmap = 'gray', vmin =0, vmax=1)
plt.savefig('poison.png')

torch.save(xtp, 'a2o/bd_inputs.pt')
torch.save(ytp, 'a2o/bd_labels.pt')

def train_model(x, y, epoch = 1000, batch_size = 5000):
    
    model = Net()
    model = model.to(device)
    indices = np.arange(len(y))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        running = 0.0
        np.random.shuffle(indices)
        for j in range(0, len(y), batch_size):
            inputs, targets = x[indices[j:j+batch_size]], y[indices[j:j+batch_size]]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
            
        print(i+1,'/', epoch, running)
    return model.cpu()

def test_model(model, x, y):
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1)
        total = pred.eq(y).sum().item()
        print(pred, y)
        print(total*1.0/len(pred), total, len(pred))

print('########Training A2O###########')
train_x, train_y = torch.cat([x, xp], 0), torch.cat([y, yp], 0)
model  = train_model(train_x, train_y)
torch.save(model.state_dict(), 'a2o/a2o.pth')
test_model(model, xt, yt)
test_model(model, xtp, ytp)
