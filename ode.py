import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("bright")
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
import torch.utils.data as data

import scipy.io as sio
from imblearn.over_sampling import SMOTE
from skimage import transform

use_cuda = torch.cuda.is_available()
train_losses = []
valid_losses = []
train_accuracy = []
valid_accuracy = []

def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

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

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

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

"""# Application

## _Learning true dynamics function (proof of concept)_
"""

class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)

class SpiralFunctionExample(LinearODEF):
    def __init__(self):
        super(SpiralFunctionExample, self).__init__(Tensor([[-0.1, -1.], [1., -0.1]]))

class RandomLinearODEF(LinearODEF):
    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2)/2.)


class TestODEF(ODEF):
    def __init__(self, A, B, x0):
        super(TestODEF, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.x0 = nn.Parameter(x0)

    def forward(self, x, t):
        xTx0 = torch.sum(x*self.x0, dim=1)
        dxdt = torch.sigmoid(xTx0) * self.A(x - self.x0) + torch.sigmoid(-xTx0) * self.B(x + self.x0)
        return dxdt

class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out

def to_np(x):
    return x.detach().cpu().numpy()

def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)

    if trajs is not None: 
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
        if save is not None:
            plt.savefig(save)
    plt.show()

def conduct_experiment(ode_true, ode_trained, n_steps, name, plot_freq=10):
    # Create data
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    t_max = 6.29*5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])

    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    obs = obs + torch.randn_like(obs) * 0.01

    # Get trajectory of random timespan 
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32
    def create_batch():
        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_

    # Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.001)
    for i in range(n_steps):
        obs_, ts_ = create_batch()

        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)

            plot_trajectories(obs=[obs], times=[times], trajs=[z_p])
            clear_output(wait=True)

ode_true = NeuralODE(SpiralFunctionExample())
ode_trained = NeuralODE(RandomLinearODEF())

func = TestODEF(Tensor([[-0.1, -0.5], [0.5, -0.1]]), Tensor([[0.2, 1.], [-1, 0.2]]), Tensor([[-1., 0.]]))
ode_true = NeuralODE(func)

func = NNODEF(2, 16, time_invariant=True)
ode_trained = NeuralODE(func)


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
#print(statarr_res)
imgarr_res = imgarr_res.reshape(-1, 1,128, 128)
#print(imgarr_res.shape)



#converts back to normal
#==============================================

#input_shape = (128, 128, 1)

#splits 20% of data into test set, other 80% into training
#x_train, x_test, y_train, y_test, vtrain, vtest = train_test_split(imgarr_res, statarr_res,numarr_res, test_size = 0.20)

img_tensor = torch.from_numpy(imgarr_res)
finished_data = ()

for i in range(img_tensor.shape[0]):
  temp = (img_tensor[i],nstat_t[i])
  finished_data = finished_data+(temp,)
# for i in finished_data:
#   print(i)
#   print(i[0].dtype)
#   break
# print(len(finished_data))


img_std = 0.3081
img_mean = 0.1307
'''
data1 = torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize((img_mean,), (img_std,))
                             ])
    )
for i in data1:
  print(i)
'''
data_train, data_test = data.random_split(finished_data, [2350, 587], generator = torch.Generator().manual_seed(42))
data_train, data_valid = data.random_split(data_train, [1880, 470], generator = torch.Generator().manual_seed(42))

data_train, data_test = data.random_split(finished_data, [2643, 293], generator = torch.Generator().manual_seed(42))
data_train, data_valid = data.random_split(data_train, [2351, 293], generator = torch.Generator().manual_seed(42))

train_loader = data.DataLoader(
    data_train,
    batch_size=128, shuffle=True
)

test_loader = data.DataLoader(
    data_test,
    batch_size=32, shuffle=True
)

valid_loader = data.DataLoader(
    data_valid,
    batch_size=32, shuffle=True
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

def valid():
    accuracy = 0.0
    num_items = 0
    total_losses = []
    model.eval()
    criterion = nn.CrossEntropyLoss()
    print(f"Validating...")
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(valid_loader),  total=len(valid_loader)):
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
    print("Valid loss: {:.3f}".format(np.mean(total_losses)))
    valid_losses.append(np.mean(total_losses))
    valid_accuracy.append(accuracy)
    return loss.item()

def test():
    accuracy = 0.0
    num_items = 0
    model.eval()
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
    accuracy = accuracy * 100 / num_items
    print("Test Accuracy: {:.3f}%".format(accuracy))

n_epochs = 250
#test()
loss = 100
temp = 0

for epoch in range(1, n_epochs + 1):
    train(epoch)
    temp = valid()
    if temp < loss:
        torch.save(model, 'best-model.pt')
        loss = temp

model = torch.load('best-model.pt')
test()

plt.plot(train_accuracy)
plt.plot(valid_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'])
plt.title('Model accuracy')

plt.savefig('ode_acc.png')

plt.clf()
plt.plot(train_losses)
plt.plot(valid_losses)
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend(['Train','Valid'])
plt.title('Model loss')
#plt.ylim(0, 1)

plt.savefig('ode_loss.png')




