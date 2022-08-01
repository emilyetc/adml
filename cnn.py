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

    print('Training... ')

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        loss_train = []
        num_items = 0
        min_loss = 100


        for imgs, labels in train_loader:
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

            loss_train += [loss.item()]
            num_items += imgs.shape[0]
            
            if loss.item() < min_loss:
                torch.save(model.state_dict(), 'cnn_model.pt')
                min_loss = loss.item()

        print('Train loss: {:.5f}'.format(np.mean(loss_train)))

    
    accuracy = 0.0
    num_items = 0

    model.eval()
    print(f"Testing...")
    val_losses = []



    loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs, labels = torch.tensor(imgs), torch.tensor(labels)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.float()
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)

            loss = criterion(output, torch.max(labels, 1)[1])
            val_losses += [loss]

            pred = torch.max(output, 1)[1].data.squeeze()

            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        loss /= len(train_loader.dataset)

        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(loss, correct, len(train_loader.dataset), 100 * correct / len(train_loader.dataset)))

            #accuracy += torch.sum(output == torch.max(labels, 1)[1]).item()
            #num_items += data.shape[0]

    #accuracy = accuracy * 100 / num_items
    #print("Test Accuracy: {:.3f}%".format(accuracy))


    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses,label="val")
    plt.plot(loss_train,label="train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("cnn.png")

        
#  plt.figure(figsize=(9, 5))
#     history = pd.DataFrame({"loss": loss_train})
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.savefig("cnn_loss.png")

        
        # if torch.eq(output, labels):
        #     num_correct += 1
        # num_samples += 1
    # accuracy = num_correct * 100 / num_samples
    # print(num_samples)
    # print(num_correct)
    # print("Test Accuracy: {:.3f}%".format(accuracy))



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



    
