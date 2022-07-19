import tensorflow
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, constant, RandomNormal
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.io as sio
import random
import sklearn as sk
import sklearn
from sklearn.metrics._ranking import precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from skimage import transform
from PIL import Image
import math
from matplotlib import pyplot

import pandas as pd
from json.decoder import NaN
from numpy.random.mtrand import randint

#parameters for the NN
epochs = 250
batch_size = 40
seed = 1
num_classes = 3 

#loading data
dataLabel = ['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','MidTemp','APOE4']#'AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','MidTemp'
stat=sio.loadmat('/content/drive/MyDrive/Colab Notebooks/stat45.mat')  #includes patient ID, classification
img=sio.loadmat('/content/drive/MyDrive/Colab Notebooks/img45.mat')    #image MRI/PET
numex = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/ADNIMERGE2.xlsx',index_col = 0,header=0)

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
                
        randStat.append(randint(0,3))#random labels created for testing purposes
        
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
print(imgarr.shape)
stat_t = []

#converts classification to 0,1,2 and appends to stat_t
#this is a modification about statarr which reads stat45. Ignore if you using new labels
for i in range(len(imgarr)):
        stat_t = np.append(stat_t, statarr[i, 0] * 0 + statarr[i, 1] * 1 + statarr[i, 2] * 2)

sm = SMOTE(sampling_strategy='all', k_neighbors = 3)

imgarr_res, nstat_t = sm.fit_sample(imgarr, statarr2)
numarr_res,xstat_t = sm.fit_sample(numarr, statarr2)
'''
comparison = nstat_t == xstat_t
equal_arrays = comparison.all()
print(equal_arrays)
'''

statarr_res = np.zeros((len(nstat_t), 3))

for i in range(len(nstat_t)):
        statarr_res[i, int(nstat_t[i])] = 1
print(statarr_res)
imgarr_res = imgarr_res.reshape(-1, 128, 128, 1)
print(imgarr_res.shape)
#converts back to normal
#==============================================

input_shape = (128, 128, 1)

#splits 20% of data into test set, other 80% into training
x_train, x_test, y_train, y_test, vtrain, vtest = train_test_split(imgarr_res, statarr_res,numarr_res, test_size = 0.20)



print("Building Model.")

#Building model begins here: see the layers below
#first branch: a cnn for images

cnninput = tensorflow.keras.layers.Input(shape=input_shape)
conv=cnninput
conv=(Conv2D(32, kernel_size = (9, 9), strides = (1, 1), activation = 'relu', padding = "same",
                 kernel_initializer = GlorotUniform(), bias_initializer = constant(0.1),
                 kernel_regularizer = l2(0.1)))(conv)

conv=(BatchNormalization())(conv)

conv=(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))(conv)

conv=(Conv2D(64, kernel_size = (9, 9), strides = (1, 1), activation = 'relu', padding = "same", 
                 kernel_initializer = GlorotUniform(), bias_initializer = constant(0.1),
                 kernel_regularizer = l2(0.1)))(conv)

conv=(BatchNormalization())(conv)

conv=(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))(conv)

conv=(Dropout(0.5))(conv)

conv=(Flatten())(conv)
          
conv=(Dense(8, activation = 'relu',  kernel_initializer = GlorotUniform(),
                kernel_regularizer = l2(0.1), bias_initializer = constant(0.1)))(conv)
cnn = tensorflow.keras.models.Model(cnninput,conv)

#second branch, an mlp for data
dim=numsize
'''
mlp = Sequential()
mlp.add(Dense(16, input_dim=dim, activation="relu"))
mlp.add(Dense(8, activation="relu"))

combine the output of those two branches
'''
#mlpinput = tensorflow.keras.layers.Input(shape = dim)
mlp = Sequential()
mlp.add(Dense(1024, input_shape=(numsize,), activation='sigmoid'))
#mlp.add(Dense(32, input_shape=(numsize,), activation='sigmoid'))
mlp.add(Dropout(0.5))


combinedInput = tensorflow.keras.layers.concatenate([mlp.output, cnn.output])
#x=Dense(516,activation="relu")(combinedInput)
x=(Dense(num_classes, activation = 'relu', kernel_initializer = GlorotUniform(),
                kernel_regularizer = l2(0.1), bias_initializer = constant(0.1)))(combinedInput)

model = tensorflow.keras.models.Model(inputs=[mlp.input, cnn.input], outputs=x)
opt = tensorflow.keras.optimizers.Adam(lr = 0.0001, epsilon = 0.05)

loss = CategoricalCrossentropy(from_logits = True)

model.compile(loss = loss, optimizer = opt, metrics = ['accuracy'])
#model.load_weights('saved_weight')

model.summary()
history = model.fit([vtrain, x_train], y_train, batch_size = batch_size, epochs = epochs,
            verbose = 1, validation_data = ([vtest, x_test], y_test), shuffle = True)
#model.save_weights('saved_weight350')
score = model.evaluate([vtest, x_test], y_test, verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
