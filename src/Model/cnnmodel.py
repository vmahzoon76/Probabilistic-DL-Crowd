import torch
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import yaml
# from torchsummary import summary
from argparse import Namespace
import numpy as np
from sklearn import metrics
from collections import Counter
import pandas as pd
from shutil import copyfile
import matplotlib as mpl

# trajectory prediction stuff
# from data_utils.get_data import get_ped_sequences
# from data_utils.collisions import collisions_plot
import math
import random

import sys
from tqdm import tqdm
import os
import cv2

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

from datetime import datetime

import pickle

device = 'cuda:0'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN_initial = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 32, kernel_size=5, stride=2),
            )
        # self.CNN_initial = torch.nn.Sequential(
        #     torch.nn.Conv1d(2, 8, kernel_size=5, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(8, 16, kernel_size=5, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(16, 32, kernel_size=5, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(32, 32, kernel_size=5, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(32, 16, kernel_size=5, stride=2),
        # )
        self.cnn1 = torch.nn.Conv1d(2, 32, kernel_size=3, stride=2)
        self.cnn2 = torch.nn.Conv1d(32, 32, kernel_size=3, stride=2)



        #self.fc3=nn.Linear(4,200)       # FFN part for z

        self.fc1= nn.Sequential(
            nn.Linear(608, 256),
        )
        self.r=nn.ReLU()

        self.final = nn.Sequential(  # FFN after combination
            nn.Linear(260, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x,z):
        #x for lidar data, z for goal and velocity
        # out=self.cnn1(x)
        # out=self.cnn2(out)
        #out = self.cnn3(out)
        out=self.CNN_initial(x)



        out=out.reshape(out.shape[0],out.shape[1]*out.shape[2])
        r=nn.ReLU()
        out=r(out)
        out=self.fc1(out)
        out= torch.cat([out,z], dim=1)    # concatenation


        out = self.final(out)  #FFN after concatenation


        return out


def get_data(data_folder):

    sample_list = [[], []]

    with open(data_folder, 'rb') as handle:
        d=pickle.load(handle)
        sample_list[0].extend(d[0])
        sample_list[1].extend(d[1])
                #x=np.array(sample_list[0])
    z=sample_list[0]
    x=[[arr[0:720],arr[720:1440]] for arr in z]
    x=np.array(x)                         # 720 lidar data
    h = [arr[1440:1444] for arr in z]
    z=np.array(h)                             # goal + v(t)

    y=np.array(sample_list[1])                     # output

    return x,z,y



def normalize_vector(vector):
    norm_vec = vector / np.sqrt(sum(np.array(vector) ** 2))
    if np.isnan(norm_vec).any():
        norm_vec[np.isnan(norm_vec)] = 0.0
    return norm_vec





def train_model(save_folder,name, num_epochs=1500,train_type='early_stopping', train_from_scratch = True):
    loss_train=[]
    loss_valid=[]
    start_time = datetime.now()
    device = 'cuda:0'
    dt = 0.10

    print('extract training trajectories', flush=True)
    x_train, z_train, y_train = get_data("Datasets/Zara2/data_for_model/lidar/Zara2_train_part_1.pkl")          # z is for the goal and velocity, x for the lidar data

    print('extract validation trajectories', flush=True)
    # size_val = int(0.9 * x_train.shape[0])
    # x_val,z_val, y_val = x_train[size_val:],z_train[size_val:], y_train[size_val:]
    # x_train,z_train, y_train = x_train[0:size_val],z_train[0:size_val], y_train[0:size_val]
    shuffled_indices = np.random.permutation(x_train.shape[0])
    percent_train = int(x_train.shape[0] * 0.9)

    train_set_indices = shuffled_indices[:percent_train]
    valid_set_indices = shuffled_indices[percent_train:]
    x_val,z_val, y_val = x_train[valid_set_indices],z_train[valid_set_indices], y_train[valid_set_indices]
    x_train,z_train, y_train = x_train[train_set_indices],z_train[train_set_indices], y_train[train_set_indices]

    print(f'Training X shape: {x_train.shape}')
    print(f'Training Y shape: {y_train.shape}')
    print()

    print(f'Validation X shape: {x_val.shape}')
    print(f'Validation Y shape: {y_val.shape}')
    print()

    x_test, z_test, y_test = get_data("Datasets/Zara2/data_for_model/lidar/Zara2_train_part_2.pkl")
    print(f'Validation X shape: {x_test.shape}')
    print(f'Validation Y shape: {y_test.shape}')
    print()

    patience =100
    if train_type == 'early_stopping':
        num_epochs =1500
        print("Training with early stopping. Max number of epochs is ", num_epochs, ".")
    else:
        print("Training for set number of ", num_epochs, " epochs.")
    learning_rate = 0.0001
    batch_size = 1024
    #hidden_dim = t_config.hidden_dim

    # initialize model
    model = CNN().to(device)
    if train_from_scratch != True:
        model.load_state_dict(torch.load("Weights/pretrained/lidar/CNN_sfm.pth"))
    # initialize loss function and optimizer (this is what is used to update the model in back propegation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lin_train_loss = criterion(torch.from_numpy(z_train[:,0:2].copy()),
                               torch.from_numpy(y_train)).detach().numpy()
    print("MSE of velocity")
    print(f'train loss {lin_train_loss:.8f}')

    lin_val_loss = criterion(torch.from_numpy(z_val[:,0:2].copy()), torch.from_numpy(y_val)).detach().numpy()
    print(f'validation loss {lin_val_loss:.8f}')
    print()

    lin_val_loss = criterion(torch.from_numpy(z_test[:, 0:2].copy()), torch.from_numpy(y_test)).detach().numpy()
    print(f'test loss {lin_val_loss:.8f}')
    print()

    # initialize best loss and best epoch
    best_loss = 1e9
    best_epoch = 0

    train_loss_lst = []

    # Train the model
    for epoch in range(num_epochs):
        print(epoch)
        # X is a torch Variable
        n_train = y_train.shape[0]
        permutation = torch.randperm(n_train)

        loss_avg = 0
        model.train()
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i + batch_size]  # size is 1024 except the last batch (which may be < 1024 )

            batch_x,batch_z, batch_y = x_train[indices],z_train[indices], y_train[indices]

            # convert to tensors
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_z = torch.from_numpy(batch_z).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            n_batch = batch_x.shape[0]  # 1024
            batch_y = batch_y.float()

            # add augmentation later

            outs = model(batch_x.float(),batch_z.float())

            # obtain the loss function
            loss = criterion(outs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item() * n_batch / n_train

        # If training with early stopping
        if train_type == 'early_stopping':
            with torch.no_grad():
                model.eval()
                # X is a torch Variable
                n_val = x_val.shape[0]  # 1793
                permutation = torch.randperm(n_val)

                val_loss_avg = 0
                for i in range(0, n_val, batch_size):
                    indices = permutation[i:i + batch_size]  # size is 32 except the last batch (which may be < 32 )

                    batch_x,batch_z, batch_y = x_val[indices],z_val[indices], y_val[indices]

                    # convert to tensors
                    batch_x = torch.from_numpy(batch_x).to(device)
                    batch_z = torch.from_numpy(batch_z).to(device)
                    batch_y = torch.from_numpy(batch_y).to(device)
                    n_batch = batch_x.shape[0]  # 32
                    batch_y = batch_y.float()

                    # add augmentation later

                    outs = model(batch_x.float(),batch_z.float())  # (32, 10, 40)

                    # obtain the loss function
                    # obtain the loss function
                    loss = criterion(outs, batch_y)

                    val_loss_avg += loss.item() * n_batch / n_val
                loss_train.append(loss_avg)
                loss_valid.append(val_loss_avg)

                if val_loss_avg < best_loss:
                    print('best so far (saving):')
                    best_loss = val_loss_avg
                    best_epoch = epoch
                    print("hi")
                    torch.save(model.state_dict(), save_folder+name+".pth")
                    print("bye")

                    print(f'New best epoch {epoch}: train loss {loss_avg:.8f}, val loss {val_loss_avg:.8f}')

                else:
                    print(f'Epoch {epoch}: train loss {loss_avg:.8f}')
                if epoch - best_epoch > patience:
                    break

        else:
            print(f'Epoch {epoch}: train loss {loss_avg:.8f}')

    print('Training duration: {}'.format(datetime.now() - start_time))

    model.load_state_dict(torch.load(save_folder+name+".pth"))
    with torch.no_grad():
        model.eval()
        # X is a torch Variable
        n_val = x_test.shape[0]  # 1793
        permutation = torch.randperm(n_val)

        val_loss_avg = 0
        for i in range(0, n_val, batch_size):
            indices = permutation[i:i + batch_size]  # size is 32 except the last batch (which may be < 32 )

            # convert to tensors
            batch_x, batch_z, batch_y = x_test[indices], z_test[indices], y_test[indices]

            # convert to tensors
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_z = torch.from_numpy(batch_z).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            n_batch = batch_x.shape[0]  # 32
            batch_y = batch_y.float()

            # add augmentation later

            outs = model(batch_x.float(),batch_z.float())  # (32, 10, 40)

            # obtain the loss function
            # obtain the loss function
            loss = criterion(outs, batch_y)

            val_loss_avg += loss.item() * n_batch / n_val
    print(f"test loss: {val_loss_avg}")
    return loss_train,loss_valid


def test_model(model, data_folder):
    # initialize loss function and optimizer (this is what is used to update the model in back propegation)
    criterion = nn.MSELoss()

    print('extract testing trajectories', flush=True)
    x_val, z_val, y_val = get_data(data_folder)
    print(f'Testing X shape: {x_val.shape}')
    print(f'Testing Y shape: {y_val.shape}')
    print()

    print("Linear model")
    lin_val_loss = criterion(torch.from_numpy(z_val[:,0:2].copy()), torch.from_numpy(y_val)).detach().numpy()
    print(f'testing loss {lin_val_loss:.8f}')
    print()

    batch_size = 1024

    with torch.no_grad():
        model.eval()
        # X is a torch Variable
        n_val = x_val.shape[0]  # 1793
        permutation = torch.randperm(n_val)

        val_loss_avg = 0
        for i in range(0, n_val, batch_size):
            indices = permutation[i:i + batch_size]  # size is 32 except the last batch (which may be < 32 )

            batch_x,batch_z, batch_y = x_val[indices], z_val[indices], y_val[indices]

            # convert to tensors
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_z = torch.from_numpy(batch_z).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            n_batch = batch_x.shape[0]  # 32
            batch_y = batch_y.float()

            # add augmentation later

            outs = model(batch_x.float(),batch_z.float())  # (32, 10, 40)

            # obtain the loss function
            # obtain the loss function
            loss = criterion(outs, batch_y)

            val_loss_avg += loss.item() * n_batch / n_val

    return val_loss_avg




if __name__ == '__main__':

    num_epochs = 1500
    train_type = 'early_stopping'  # 'overfitting'
    train_arr, loss_arr = train_model("Weights/Zara2/lidar/finetuning/","finetuning_zara2_1", num_epochs, train_from_scratch=False)




