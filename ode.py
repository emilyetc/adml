import math
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import nn
import scipy.io as sio
from imblearn.over_sampling import SMOTE
from skimage import transform

use_cuda = torch.cuda.is_available()
train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

def norm(dim):
    return nn.BatchNorm2d(dim)

def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)

def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)

class ConvODEF(ODEF):
    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.conv1 = conv3x3(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv3x3(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt

class ContinuousNeuralMNISTClassifier(nn.Module):
    def __init__(self, ode):
        super(ContinuousNeuralMNISTClassifier, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ode
        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.feature(x)
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out

func = ConvODEF(64)
ode = NeuralODE(func)
model = ContinuousNeuralMNISTClassifier(ode)
if use_cuda:
    model = model.cuda()

dataLabel = ['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','MidTemp','APOE4']#'AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','MidTemp'
stat=sio.loadmat('/sdf/home/e/ewang/suli/MNN/data/stat45.mat')  #includes patient ID, classification
img=sio.loadmat('/sdf/home/e/ewang/suli/MNN/data/img45.mat')    #image MRI/PET
numex = pd.read_excel('/sdf/home/e/ewang/suli/MNN/data/ADNIMERGE2.xlsx',index_col = 0,header=0)

todel=['__version__','__header__','__globals__'] #getting rid of empty/nonrelevant headers
for t in todel:
    del img[t]
    del stat[t]

print("Loaded Data.")
numsize =7
imgarr = np.zeros((len(img), 128, 128)) #placeholders
statarr = np.zeros((len(stat), 3))    #diagnose from stats45. ignore if you are using new labels
numarr = np.zeros((len(stat),numsize))#biological informations from adnimerge
statarr2 =[] #array for the last diagnose
randStat = [] #random
count = 0

#moving images from img to imgarr
#moving stats from stat to statarr

for i in img.keys():
        imgarr[count] = img[i]
        statarr[count] = stat[i]
        numtemp = []
        pid = i[0:10]
        rid = i[6:10]
        rid = int(rid)
        apoesum = 0
        
        stage = numex['DX_bl'][pid]
        if stage == 'CN':
            statarr2.append(0)
        elif stage == 'AD':
            statarr2.append(2)
        else:
            statarr2.append(1)
                
        #randStat.append(randint(0,3))#random labels created for testing purposes
        
        for j in dataLabel:
            if math.isnan(numex[j][pid]):
                if numex['DX_bl'][pid]=='AD':
                    numtemp.append(numex[j]['ADAVERAGE'])
                elif numex['DX_bl'][pid]=='CN':
                    numtemp.append(numex[j]['CNAVERAGE'])
                else:
                    numtemp.append(numex[j]['MCIAVERAGE'])#CHANGE TO AVERAGE VALUE MAYBE
            else:
                numtemp.append(numex[j][pid])
        numarr[count] = numtemp
        #print(numtemp)
        #print(pid,numtemp)
        count += 1
print("passed!")
for i in range(len(imgarr)):
        xmin = 127
        xmax = 0
        ymin = 127
        ymax = 0
        #finding borders of pixels with value >= threshold
        for j in range(128):
                for k in range(128):
                        if imgarr[i, j, k] >= 5e-2:
                                xmin = min(xmin, j)
                                xmax = max(xmax, j)
                                ymin = min(ymin, k)
                                ymax = max(ymax, k)
        delta = 10
        xmin = max(0, xmin - delta)
        xmax = min(127, xmax + delta)
        ymin = max(0, ymin - delta)
        ymax = min(127, ymax + delta)
        if xmin < xmax and ymin < ymax:
                imgarr[i] = transform.resize(imgarr[i, (xmin):(xmax+1), (ymin):(ymax+1)], (128, 128))


#above step reshapes all images to 128x128 focused on pixels of interest

#============================================
#gets arrays ready for smote
#flattens 128x128 pixels into 128*128 array
'''
for i in imgarr:
    plt.imshow(i)
    plt.show()
'''
imgarr = imgarr.reshape((len(imgarr), 128 * 128))
stat_t = []

#converts classification to 0,1,2 and appends to stat_t
#this is a modification about statarr which reads stat45. Ignore if you using new labels
for i in range(len(imgarr)):
        stat_t = np.append(stat_t, statarr[i, 0] * 0 + statarr[i, 1] * 1 + statarr[i, 2] * 2)

sm = SMOTE(sampling_strategy='all', k_neighbors = 10)

imgarr_res, nstat_t = sm.fit_resample(imgarr, statarr2)
numarr_res,xstat_t = sm.fit_resample(numarr, statarr2)
'''
comparison = nstat_t == xstat_t
equal_arrays = comparison.all()
print(equal_arrays)
'''

statarr_res = np.zeros((len(nstat_t), 3))

for i in range(len(nstat_t)):
        statarr_res[i, int(nstat_t[i])] = 1
print(statarr_res)
imgarr_res = imgarr_res.reshape(-1, 1,128, 128)
print(imgarr_res.shape)



#converts back to normal
#==============================================

#input_shape = (128, 128, 1)

#splits 20% of data into test set, other 80% into training
#x_train, x_test, y_train, y_test, vtrain, vtest = train_test_split(imgarr_res, statarr_res,numarr_res, test_size = 0.20)

from scipy.sparse import data
img_tensor = torch.from_numpy(imgarr_res)
finished_data = ()

for i in range(img_tensor.shape[0]):
  temp = (img_tensor[i],nstat_t[i])
  finished_data = finished_data+(temp,)
for i in finished_data:
  #print(i)
  #print(i[0].dtype)
  break
#print(len(finished_data))


batch_size = 32
train_loader = torch.utils.data.DataLoader(
    finished_data,
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    finished_data,
    batch_size=128, shuffle=True
)

optimizer = torch.optim.Adam(model.parameters())

def train(epoch):
    num_items = 0
    accuracy = 0.0
    total_losses = []

    model.train()
    criterion = nn.CrossEntropyLoss()
    print(f"Training Epoch {epoch}...")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data = data.float()
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) 
        loss.backward()
        optimizer.step()
        total_losses += [loss.item()]
        num_items += data.shape[0]
        accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
    accuracy = accuracy * 100 / num_items
    print('Train loss: {:.5f}'.format(np.mean(total_losses)))
    train_losses.append(np.mean(total_losses))
    train_accuracy.append(accuracy)

def test():
    accuracy = 0.0
    num_items = 0
    total_losses = []
    model.eval()
    criterion = nn.CrossEntropyLoss()
    print(f"Testing...")
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader),  total=len(test_loader)):
            data = data.float()
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
            num_items += data.shape[0]
            loss = criterion(output, target) 
            total_losses += [loss.item()]
    accuracy = accuracy * 100 / num_items
    print("Test Accuracy: {:.3f}%".format(accuracy))
    test_losses.append(np.mean(total_losses))
    test_accuracy.append(accuracy)

n_epochs = 250
#test()

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.title('Model accuracy')

plt.savefig('ode_acc.png')

plt.clf()
plt.plot(train_losses)
plt.plot(test_losses)
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend(['Train','Test'])
plt.title('Model loss')

plt.savefig('ode_loss.png')

