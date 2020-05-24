import os, sys, time
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from imports.utils import *
from imports.models import *
from imports.adabound import AdaBound 

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import itertools

X_train_arr = np.load('Your file path for training datasets of X_train') # recommenended total batch size is over than 500,000 samples
Y_train_arr = np.load('Your file path for training datasets of Y_train')
X_valid_arr = np.load('Your file path for training datasets of X_valid')
Y_valid_arr = np.load('Your file path for training datasets of Y_valid')

X_train_arr = np.expand_dims(X_train_arr, axis=-2)
Y_train_arr = np.expand_dims(Y_train_arr, axis=-2)
X_valid_arr = np.expand_dims(X_valid_arr, axis=-2)
Y_valid_arr = np.expand_dims(Y_valid_arr, axis=-2)

CUDA_DEVICE = 0
torch.cuda.set_device(device=CUDA_DEVICE) 
    
model = Denoise_Model().cuda()
try:
    pred = model(torch.Tensor(X_train_arr[:128]).cuda())
    print(pred.shape) 
except:
    raise NameError("The input shape should be revised")
total_parameter = sum(p.numel() for p in model.parameters()) 
print('total_parameter: ', total_parameter / 1000000, 'M')

SAVE_DIR = './data/models/'
filename = 'denoising_model.pt'
epochs = 30
train_batch = X_train_arr.shape[0]
mini_batch = 128
valid_mini_batch = 128
learning_rate = 0.001
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss().cuda()

total_loss = []
total_val_loss = []
for epoch in range(epochs):
    model.train() 
    avg_cost = 0
    tic = time.time() 

    for i in range(train_batch // mini_batch):
        train_indices = np.random.choice(train_batch, size=mini_batch)
        x_train_temp = torch.Tensor(X_train_arr[train_indices]).cuda()
        y_train_temp = torch.Tensor(Y_train_arr[train_indices]).cuda()

        optimizer.zero_grad() 
        hypothesis = model(x_train_temp) 
        cost = criterion(hypothesis, y_train_temp) 
        cost.backward() 
        optimizer.step() 

        avg_cost += cost
        print(round(((i+1)*mini_batch)/train_batch*100), '% in Epoch', end='\r')
    loss_temp = avg_cost / (train_batch // mini_batch)
    total_loss.append(loss_temp.cpu().detach().item())
    print("Epoch:", '%4d' % (epoch + 1), ' | Loss =', '{:.5f}'.format(loss_temp), end=' | ')
    with torch.no_grad():
        model.eval()
        valid_indices = torch.randperm(X_valid_arr.shape[0])[:valid_mini_batch]
        X_valid = torch.Tensor(X_valid_arr[valid_indices]).cuda()
        Y_valid = torch.Tensor(Y_valid_arr[valid_indices]).cuda()

        prediction = model(X_valid)
        val_loss = criterion(prediction, Y_valid)

        print('Val_loss: {:.5f}'.format(val_loss.item()))
        total_val_loss.append(val_loss.cpu().detach().item())
    total_val_loss = np.array(total_val_loss)
    if total_val_loss.min() >= total_val_loss[-1]:
        torch.save(model.state_dict(), SAVE_DIR+filename)
    else:
        if np.min(total_val_loss[-3:-1]) < total_val_loss[-1]:
            learning_rate *= 0.5
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("lr: ", learning_rate)
    total_val_loss = list(total_val_loss) 
