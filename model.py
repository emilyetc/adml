import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import collections.abc
collections.Iterable = collections.abc.Iterable
from torchsample.modules import ModuleTrainer
import matplotlib.pyplot as plt
from itertools import chain

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #CNN
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size=(9, 9), stride = (1, 1), padding = 'same')
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias.data, 0.1)
        self.bn1 = nn.BatchNorm2d(7)
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding = 'same')
        
        self.conv2 = nn.Conv2d(in_channels= 1, out_channels= 64, kernel_size=(9, 9), stride = (1, 1), padding = 'same')
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias.data, 0.1)
        self.bn2 = nn.BatchNorm2d(7)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding = 'same')
        self.drop1 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features = 7, out_features = 8)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias.data, 0.1)

        #MLP
        self.fc2 = nn.Linear(in_features = 7, out_features = 1024)

        #Concatenate
        self.fc3 = nn.Linear(in_features = 2, out_features = 3)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias.data, 0.1)



    def forward(self, x_train, y_train):
        #CNN
        x = self.conv1(x_train)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.drop1(x)

        x = nn.Flatten(x)
        x = self.fc1(x)
        x = F.relu3(x)

        x = x.view(-1, 64*64)

        #MLP
        y = self.fc2(y_train)
        y = F.sigmoid(y)
        y = self.drop1(y)

        #concatenate
        combined = torch.cat((y, x), 1)
        combined = self.fc3(combined)
        combined = F.relu(combined)

        return combined


if __name__ == '__main__':

    print("loading data...")
    statarr_res = np.load(r'data/statarr_res.npy')
    imgarr_res = np.load(r'data/imgarr_res.npy')
    numarr_res = np.load(r'data/numarr_res.npy')
    print("data loaded")

    print("splitting data...")
    x_train, x_test = data.random_split(imgarr_res, [587, 2350], generator = torch.Generator().manual_seed(42))
    y_train, y_test = data.random_split(statarr_res, [587, 2350], generator = torch.Generator().manual_seed(42))
    z_train, z_test = data.random_split(numarr_res, [587, 2350], generator = torch.Generator().manual_seed(42))
    print("data split")
    
    model = Network()
    trainer = ModuleTrainer(model)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, eps = 0.05)
    criterion = nn.CrossEntropyLoss()

    trainer.compile(loss = criterion, optimizer = optimizer)

    
    trainer.fit([z_train, x_train], y_train, batch_size = 40, num_epoch = 250)

    score = model.evaluate([z_test, x_test], y_test, verbose = 0)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # # Plot training & validation accuracy values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    