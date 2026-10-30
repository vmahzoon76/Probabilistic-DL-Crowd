import torch
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import yaml
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


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, inputs, pred_type, out_dim, n_objects, nhead, nhid, nlayers, dropout=0.0):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.nhid = nhid
        self.inputs = inputs
        self.out_dim = out_dim
        encoder_layers = TransformerEncoderLayer(nhid, nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # ADDING LAYERS
        # change each split from last dimension 20 to 64 with linear layer
        self.fc1 = nn.Sequential(  # sequential operation
            nn.Linear(inputs, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(), )

        self.fc_roles = nn.Sequential(  # sequential operation
            nn.Linear(inputs, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(), )

        # trajectory prediction layers
        self.out = nn.Sequential(  # sequential operation
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, out_dim), )

    def forward(self, src):
        # Uncomment to remove the intent from the src

        (P, B, E) = src.shape  # (32, 11, 6)
        # ADDING LAYERS
        src = src.permute(1, 0, 2)  # (11, 32, 6)   # why 32 ? why 11 at the begiinning and why not 66 at end
        (P, B, E) = src.shape

        # Uncomment for feed forward
        src = self.fc1(src)  # (replace with LSTM)

        # add role embedding (ADDITIVE. NOT CONCAT)
        '''
        A_embed = torch.zeros(2).float()
        A_embed[-2] = 1
        A_embed = self.fc_roles(output_A.to(device))

        other_embed = torch.zeros(2).float()
        other_embed[-1] = 1
        other_embed = self.fc_roles(output_other.to(device))
        '''

        output = self.transformer_encoder(src)  # [11, 32, 128]

        # use torch.transpose to change to dim from (11, 32, 128) to (32, 11, 128)
        output = torch.transpose(output, 0, 1)

        '''
        output_A = output[:,0:1,:]
        output_other = output[:,1:,:].sum(1).reshape(B, 1, output.shape[-1])
        output = torch.add(output_A, output_other)
        '''

        output = self.out(output[:, 0:1, :])  # [32, 1, 2]

        return output  # [32, 1, 2]


def get_data(data_folder, max_ped):
    x = []
    y = []

    with open(data_folder, 'rb') as handle:
        sample_list = pickle.load(handle)

    for sample_idx in range(len(sample_list)):
        this_sample = sample_list[sample_idx]

        curr_loc_df = this_sample[0][["curr_loc_x", "curr_loc_y"]]
        curr_vel_df = this_sample[0][["curr_vel_x", "curr_vel_y"]]
        pref_vel_df = this_sample[0][["goal_loc_x", "goal_loc_y"]]

        pref_vel_df = pref_vel_df - np.array(curr_loc_df.iloc[0])
        curr_loc_df = curr_loc_df - curr_loc_df.iloc[0]

        # reshape the arrays into (#sample, #agents, #components)
        (s, c) = curr_loc_df.shape
        x_curr_loc = np.concatenate(
            (np.array(curr_loc_df).reshape(1, s, c), np.ones((1, max_ped - s, 2)) * 999), axis=1)
        x_curr_vel = np.concatenate(
            (np.array(curr_vel_df).reshape(1, s, c), np.ones((1, max_ped - s, 2)) * 999), axis=1)
        x_pref_vel = np.concatenate(
            (np.array(pref_vel_df).reshape(1, s, c), np.ones((1, max_ped - s, 2)) * 999), axis=1)

        x_combined = np.concatenate((x_curr_loc, x_curr_vel, x_pref_vel), axis=-1)  # (1, 11, 6)

        y_curr_vel = np.array(this_sample[1][["pred_vel_x", "pred_vel_y"]]).reshape(1, 1, 2)

        # append the sample to the x and y lists
        x.append(x_combined)
        y.append(y_curr_vel)
    #     print(x.shape)
    #     # turn the lists into arrays. Then return arrays
    x = np.vstack(x)  # (#numsamples, 11, 6)
    y = np.vstack(y)  # (#numsamples, 11, 2)

    return x, y


def update_df(df, step, data_list):
    df.loc[step] = data_list
    return df


def normalize_vector(vector):
    norm_vec = vector / np.sqrt(sum(np.array(vector) ** 2))
    if np.isnan(norm_vec).any():
        norm_vec[np.isnan(norm_vec)] = 0.0
    return norm_vec


def train_model(dataset,save_folder, name, train_type='early_stopping', train_from_scratch=True):
    train_loss = []
    valid_loss = []
    start_time = datetime.now()
    device = 'cuda:0'
    dt = 0.10
    datasets=["Hotel","ETH","Zara1","Zara2"]
    print('extract training trajectories', flush=True)
    print()
    all_training_data=[]
    for data in datasets:
        if data!=dataset:
            all_training_data.append(get_data("Datasets/"+data+"/data_for_model/transformer/whole_"+data.lower()+".pkl", 27))
            print(all_training_data[-1][0].shape)
            print(f"{data} added in train")
        else:
            x_test, y_test = get_data("Datasets/"+data+"/data_for_model/transformer/whole_"+data.lower()+".pkl", 27)
            print(f"{data} added in test")

    x_train=np.r_[all_training_data[0][0],all_training_data[1][0],all_training_data[2][0]]
    y_train=np.r_[all_training_data[0][1],all_training_data[1][1],all_training_data[2][1]]


    #x_train = np.r_[all_training_data[0][0],all_training_data[1][0]]
    #y_train=np.r_[all_training_data[0][1],all_training_data[1][1]]
    shuffled_indices = np.random.permutation(x_train.shape[0])
    percent_train = int(x_train.shape[0] * 0.9)

    train_set_indices = shuffled_indices[:percent_train]
    valid_set_indices = shuffled_indices[percent_train:]
    x_val, y_val = x_train[valid_set_indices], y_train[valid_set_indices]
    x_train, y_train = x_train[train_set_indices], y_train[train_set_indices]



    print(f'Training X shape: {x_train.shape}')
    print(f'Training Y shape: {y_train.shape}')
    print('extract validation trajectories', flush=True)
    print(f'Validation X shape: {x_val.shape}')
    print(f'Validation Y shape: {y_val.shape}')
    print()
    print(f'test X shape: {x_test.shape}')
    print(f'test Y shape: {y_test.shape}')
    print()

    # uses the transformer yaml file to get some variables
    num_epochs = 500
    patience = 10
    if train_type == 'early_stopping':
        print("Training with early stopping. Max number of epochs is ", num_epochs, ".")
    else:
        print("Training for set number of ", num_epochs, " epochs.")
    learning_rate = float(0.0005)
    batch_size = 1024  # 1024

    # initialize model
    model = TransformerModel(inputs=6, pred_type="trajectory", out_dim=2, n_objects=27, nhead=8, nhid=128,
                             nlayers=4).to(device)

    if train_from_scratch != True:
        model.load_state_dict(torch.load("Weights/pretrained/transformer/RVO_DSFM_500000.pth"))

    # for param in model.transformer_encoder.parameters():
    #     param.requires_grad = False
    #
    # # Freeze fc1 layers
    # for param in model.fc1.parameters():
    #     param.requires_grad = False
    # initialize loss function and optimizer (this is what is used to update the model in back propegation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lin_train_loss = criterion(torch.from_numpy(x_train[:, 0:1, 2:4].copy()),
                               torch.from_numpy(y_train)).detach().numpy()
    print("MSE of velocity")
    print(f'train loss {lin_train_loss:.8f}')

    lin_val_loss = criterion(torch.from_numpy(x_val[:, 0:1, 2:4].copy()), torch.from_numpy(y_val)).detach().numpy()
    print(f'validation loss {lin_val_loss:.8f}')
    print()

    lin_val_loss = criterion(torch.from_numpy(x_test[:, 0:1, 2:4].copy()), torch.from_numpy(y_test)).detach().numpy()
    print(f'test loss {lin_val_loss:.8f}')
    print()

    # initialize best loss and best epoch
    best_loss = 1e9
    best_epoch = 0

    train_loss_lst = []

    # Train the model
    for epoch in range(num_epochs):
        # X is a torch Variable
        n_train = y_train.shape[0]
        permutation = torch.randperm(n_train)

        loss_avg = 0
        model.train()
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i + batch_size]  # size is 1024 except the last batch (which may be < 1024 )

            batch_x, batch_y = x_train[indices], y_train[indices]

            # convert to tensors
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            n_batch = batch_x.shape[0]  # 1024
            batch_y = batch_y.float()

            # add augmentation later

            outs = model(batch_x.float())

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

                    batch_x, batch_y = x_val[indices], y_val[indices]

                    # convert to tensors
                    batch_x = torch.from_numpy(batch_x).to(device)
                    batch_y = torch.from_numpy(batch_y).to(device)
                    n_batch = batch_x.shape[0]  # 32
                    batch_y = batch_y.float()

                    # add augmentation later

                    outs = model(batch_x.float())  # (32, 10, 40)

                    # obtain the loss function
                    # obtain the loss function
                    loss = criterion(outs, batch_y)

                    val_loss_avg += loss.item() * n_batch / n_val
                valid_loss.append(val_loss_avg)
                train_loss.append(loss_avg)
                if val_loss_avg < best_loss:
                    print('best so far (saving):')
                    best_loss = val_loss_avg
                    best_epoch = epoch
                    print("hi")
                    torch.save(model.state_dict(), save_folder + name + ".pth")
                    print("bye")

                    print(f'New best epoch {epoch}: train loss {loss_avg:.8f}, val loss {val_loss_avg:.8f}')

                else:
                    print(f'Epoch {epoch}: train loss {loss_avg:.8f}')
                if epoch - best_epoch > patience:
                    break

        else:
            print(f'Epoch {epoch}: train loss {loss_avg:.8f}')

    model.load_state_dict(torch.load(save_folder + name + ".pth"))
    with torch.no_grad():
        model.eval()
        # X is a torch Variable
        n_val = x_test.shape[0]  # 1793
        permutation = torch.randperm(n_val)

        val_loss_avg = 0
        for i in range(0, n_val, batch_size):
            indices = permutation[i:i + batch_size]  # size is 32 except the last batch (which may be < 32 )

            batch_x, batch_y = x_test[indices], y_test[indices]

            # convert to tensors
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            n_batch = batch_x.shape[0]  # 32
            batch_y = batch_y.float()

            # add augmentation later

            outs = model(batch_x.float())  # (32, 10, 40)

            # obtain the loss function
            # obtain the loss function
            loss = criterion(outs, batch_y)

            val_loss_avg += loss.item() * n_batch / n_val
    print(f"test loss: {val_loss_avg}")
    return train_loss, valid_loss


if __name__ == '__main__':
    # train
        train_loss, valid_loss = train_model("Hotel","Weights/Hotel/transformer/finetuning/", "finetuning_hotel_leave_one_out", train_from_scratch=False)

