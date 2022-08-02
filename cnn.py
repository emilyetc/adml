import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import chain
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #CNN
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size=(9, 9), stride = (1, 1), padding = 'same')
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias.data, 0.1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=(9, 9), stride = (1, 1), padding = 'same')
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias.data, 0.1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.drop1 = nn.Dropout(0.5)
        self.flatten1 = nn.Flatten()

        self.fc1 = nn.Linear(in_features = 65536, out_features = 3)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias.data, 0.1)


    def forward(self, imgs):
        #CNN
        x = self.conv1(imgs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.drop1(x)

        x = self.flatten1(x)
        x = self.fc1(x)

        return x

train_accu = []
train_losses = []


def train(epoch):
    print('\nEpoch : %d'%epoch)

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = torch.tensor(imgs), torch.tensor(labels)
        imgs = imgs.permute(0, 3, 1, 2)
        model.train()
        imgs = imgs.float()
        imgs = imgs.to(device)
        labels = labels.to(device)

        output = model(imgs)

        loss = criterion(output, torch.max(labels, 1)[1])

        l2_lambda = 0.1
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += torch.sum(torch.max(output, dim=1).indices == torch.max(labels, dim=1).indices)

    train_loss = running_loss / len(train_loader)
    accu = 100. * float(correct) / total
    train_accu.append(accu)
    train_losses.append(train_loss)

    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

eval_losses=[]
eval_accu=[]

def test(epoch):
    model.eval()

    running_loss=0
    correct=0
    total=0
    
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs, labels = torch.tensor(imgs), torch.tensor(labels)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.float()
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)

            loss = criterion(output, torch.max(labels, 1)[1])

    
            running_loss+=loss.item()

            total += labels.size(0)
            correct += torch.sum(torch.max(output, dim=1).indices == torch.max(labels, dim=1).indices)
    
    test_loss=running_loss/len(test_loader)
    accu=100.*float(correct)/total
 
    eval_losses.append(test_loss)
    eval_accu.append(accu)
    
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 





if __name__ == '__main__':

    print('Loading Data...')

    imgs = np.load('data/imgarr_res.npy')
    labels = np.load('data/statarr_res.npy')

    tot_data = []
    for i in range(len(imgs)):
        tot_data.append([imgs[i], labels[i]])
    
    data_train, data_test = data.random_split(tot_data, [2350, 587], generator = torch.Generator().manual_seed(42))

    train_loader = data.DataLoader(data_train, batch_size=32)
    test_loader = data.DataLoader(data_test, batch_size=32)
    print('Data Loaded')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Network()
    model = model.to(device)
    epochs = 250

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, eps = 0.05)
    criterion = nn.CrossEntropyLoss()



    for epoch in range(1, epochs+1):
        train(epoch)
        test(epoch)

    print(train_accu)
    print(train_losses)
    
    plt.plot(train_accu)
    plt.plot(eval_accu)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.title('Model accuracy')
    
    plt.savefig('cnn_acc.png')

    plt.clf()
    plt.plot(train_losses)
    plt.plot(eval_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(['Train','Test'])
    plt.title('Model loss')
    
    plt.savefig('cnn_loss.png')

