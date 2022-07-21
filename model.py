import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import chain

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

        self.fc1 = nn.Linear(in_features = 65536, out_features = 8)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias.data, 0.1)

        #MLP portion
        self.fc2 = nn.Linear(in_features = 7, out_features = 1024)

        #Concatenate
        self.fc3 = nn.Linear(in_features = 1032, out_features = 3)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias.data, 0.1)



    def forward(self, imgs, feats):
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
        x = F.relu(x)

        #x = x.view(-1, 64*64)

        #MLP
        y = self.fc2(feats)
        y = F.sigmoid(y)
        y = self.drop1(y)

        #Concatenate
        combined = torch.cat((y, x), 1)
        combined = self.fc3(combined)
        combined = F.relu(combined)

        return combined

class CombinedData(data.Dataset):
    def __init__(self, train=True, seed=42):
        self.is_train = train

        feats = np.load('data/numarr_res.npy')
        imgs = np.load('data/imgarr_res.npy')
        labels = np.load('data/statarr_res.npy')

        triples = list(zip(feats, imgs, labels))
        self.train, self.test = data.random_split(triples, [2350, 587], generator=torch.Generator().manual_seed(seed))

    def __getitem__(self, i):
        return self.train[i] if self.is_train else self.test[i]

    def __len__(self):
        return len(self.train) if self.is_train else len(self.test)

if __name__ == '__main__':

    # print("loading data...")
    # statarr_res = np.load(r'data/statarr_res.npy')
    # imgarr_res = np.load(r'data/imgarr_res.npy')
    # numarr_res = np.load(r'data/numarr_res.npy')
    # print("data loaded")

    # print("splitting data...")
    # img_train, img_test = data.random_split(imgarr_res, [2350, 587], generator = torch.Generator().manual_seed(42))
    # stat_train, stat_test = data.random_split(statarr_res, [2350, 587], generator = torch.Generator().manual_seed(42))
    # num_train, num_test = data.random_split(numarr_res, [2350, 587], generator = torch.Generator().manual_seed(42))
    # print("data split")
    

    # train_loader = data.DataLoader(ConcatDataset([num_train, img_train], stat_train), batch_size = 40)
    # test_loader = data.DataLoader(ConcatDataset([num_test, img_test], stat_test), batch_size = 40)
    
    print('Loading Data...')
    train_dataset = CombinedData(train=True)
    test_dataset = CombinedData(train=False)

    train_loader = data.DataLoader(train_dataset, batch_size=32)
    print('Data Loaded')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Network()
    model = model.to(device)
    epochs = 250

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, eps = 0.05)
    criterion = nn.CrossEntropyLoss()

    i = 0
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        loss_train = 0.0
        for feats, imgs, labels in train_loader:
            feats, imgs, labels = torch.tensor(feats), torch.tensor(imgs), torch.tensor(labels)
            imgs = imgs.permute(0, 3, 1, 2)
            model.train()
            imgs = imgs.float()
            feats = feats.float()
            imgs = imgs.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            output = model(imgs, feats)

            loss = criterion(output, labels)

            l2_lambda = 0.1
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
    


    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # print("plotting")
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



    
