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
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)

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


class ConditionalRQCoupling(nn.Module):
    def __init__(self, hidden_dim, K=16, bound=5.0, parity=0):
        super().__init__()
        self.K = K
        self.bound = bound
        self.parity = parity

        self.nn = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3*K + 1),
        )

    def forward(self, v, h, inverse=False):
        # Split velocity
        if self.parity == 0:
            v_id = v[:, :1]
            v_tr = v[:, 1:2]
        else:
            v_id = v[:, 1:2]
            v_tr = v[:, :1]

        # --- CONDITION ONLY ON h (correct) ---
        nn_input = torch.cat([v_id, h], dim=-1) 
        params = self.nn(nn_input)           # <--- FIXED
        K = self.K

        un_w = params[:, :K]
        un_h = params[:, K:2*K]
        un_d = params[:, 2*K:3*K+1]

        # Inputs must be 1D
        vtr_scalar = v_tr.squeeze(-1)

        eps = 1e-3
        # if not inverse:
        #     vtr_scalar = torch.clamp(vtr_scalar, -self.bound+eps, self.bound-eps)


        vtr_out, logdet = unconstrained_rational_quadratic_spline(
            inputs=vtr_scalar,
            unnormalized_widths=un_w,
            unnormalized_heights=un_h,
            unnormalized_derivatives=un_d,
            inverse=inverse,
            tails="linear",
            tail_bound=self.bound,
        )

        vtr_out = vtr_out.unsqueeze(-1)

        # Reassemble output
        if self.parity == 0:
            v_out = torch.cat([v_id, vtr_out], dim=-1)
        else:
            v_out = torch.cat([vtr_out, v_id], dim=-1)

        return v_out, logdet



class VelocityFlow(nn.Module):
    """
    Multi-layer conditional normalizing flow on v ∈ R^2,
    built from stacked ConditionalRQCoupling layers with alternating parity.
    """

    def __init__(self, hidden_dim, K=16, bound=5.0, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            parity = i % 2  # 0,1,0,1,...
            layers.append(ConditionalRQCoupling(hidden_dim, K, bound, parity))
        self.layers = nn.ModuleList(layers)

    def forward(self, v, h, inverse=False):
        """
        v: [B, 2]
        h: [B, hidden_dim]
        Returns:
            v_out: [B, 2]
            logdet_total: [B]
        """
        logdet_total = 0.0

        if not inverse:
            # data -> latent
            for layer in self.layers:
                v, logdet = layer(v, h, inverse=False)
                logdet_total = logdet_total + logdet
        else:
            # latent -> data (reverse order)
            for layer in reversed(self.layers):
                v, logdet = layer(v, h, inverse=True)
                logdet_total = logdet_total - logdet

        return v, logdet_total












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

        # Multi-layer conditional normalizing flow over 2D velocity
        self.flow = VelocityFlow(
            hidden_dim=nhid,
            K=8,
            bound=5.0,
            num_layers=6,     # you can tune this (4, 6, 8, ...)
        )
        self.latent_dim = 2


    def forward(self, src, mode="sample", target_v=None):
        # src: [B, P, E] -> same as before
        (P, B, E) = src.shape  
        src = src.permute(1, 0, 2)
        src = self.fc1(src)
        output = self.transformer_encoder(src)
        output = output.transpose(0, 1)  # [B, P, 128]

        h = output[:, 0, :]  # context vector [B, hidden_dim]

        if mode == "log_prob":
            # invert flow
            v =  target_v[:, 0, :] 
            z, logdet = self.flow(v, h, inverse=False)
            D = z.size(-1)
            log_p_z = -0.5 * (z**2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

            return (log_p_z + logdet).mean()

        if mode == "sample":
            B = src.size(1)
            temp = 1
            z = torch.randn(B, 2, device=src.device)*temp
            max_speed = 3.0   # choose based on real motion stats
            v, _ = self.flow(z, h, inverse=True)
            v = torch.clamp(v, min=-max_speed, max=max_speed)
            return v.view(B, 1, 2)




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


def negative_log_likelihood(mu, sigma, target):
    """
    Compute the negative log likelihood loss given mu, sigma, and the target.
    """
    eps = 1e-3  # for numerical stability
    distribution = torch.distributions.Normal(mu, sigma + eps)
    log_prob = distribution.log_prob(target)
    return -log_prob.mean()



def train_model(save_folder,name,train_type='early_stopping', train_from_scratch = True):
    train_loss=[]
    valid_loss=[]
    start_time = datetime.now()
    device = 'cuda:0'
    dt = 0.10

    criterion = negative_log_likelihood
    mse_criterion = nn.MSELoss()


    print('extract training trajectories', flush=True)
    print()
    x_train,y_train=get_data("Datasets/ETH/data_for_model/transformer/eth_train_part_1.pkl",27)
    # size_val=int(0.8*x_train.shape[0])
    # x_val, y_val = x_train[size_val:], y_train[size_val:]
    # x_train,y_train=x_train[0:size_val],y_train[0:size_val]
    shuffled_indices = np.random.permutation(x_train.shape[0])
    percent_train = int(x_train.shape[0] * 0.9)

    train_set_indices = shuffled_indices[:percent_train]
    valid_set_indices = shuffled_indices[percent_train:]
    x_val, y_val = x_train[valid_set_indices], y_train[valid_set_indices]
    x_train,y_train=x_train[train_set_indices],y_train[train_set_indices]

    x_test,y_test=get_data("Datasets/Zara1/data_for_model/transformer/Zara1_train_part_2.pkl",27)

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
    patience =50
    if train_type == 'early_stopping':
        print("Training with early stopping. Max number of epochs is ", num_epochs, ".")
    else:
        print("Training for set number of ", num_epochs, " epochs.")
    learning_rate = float(0.0005)
    batch_size = 1024  # 1024

    # initialize model
    model = TransformerModel(inputs=6, pred_type="trajectory", out_dim=2 , n_objects=27, nhead=8, nhid=128,
                             nlayers=4).to(device)
    
    if train_from_scratch != True:
        model.load_state_dict(torch.load("Weights/pretrained/transformer/god.pth"))
    # initialize loss function and optimizer (this is what is used to update the model in back propegation)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lin_train_loss = mse_criterion(torch.from_numpy(x_train[:, 0:1, 2:4].copy()),
                               torch.from_numpy(y_train)).detach().numpy()
    print("MSE of velocity")
    print(f'train loss {lin_train_loss:.8f}')

    lin_val_loss = mse_criterion(torch.from_numpy(x_val[:, 0:1, 2:4].copy()), torch.from_numpy(y_val)).detach().numpy()
    print(f'validation loss {lin_val_loss:.8f}')
    print()

    lin_val_loss = mse_criterion(torch.from_numpy(x_test[:, 0:1, 2:4].copy()), torch.from_numpy(y_test)).detach().numpy()
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
        mse_loss_avg = 0
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

            log_prob = model(batch_x.float(), mode="log_prob", target_v=batch_y)
            loss = -log_prob


           

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            loss_avg += loss.item() * batch_x.shape[0] / n_train
        

        # If training with early stopping
        if train_type == 'early_stopping':
            with torch.no_grad():
                model.eval()
                n_val = x_val.shape[0]
                permutation = torch.randperm(n_val)

                val_loss_avg = 0
                val_mse_loss_avg = 0
                for i in range(0, n_val, batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x, batch_y = x_val[indices], y_val[indices]

                    batch_x = torch.from_numpy(batch_x).to(device)
                    batch_y = torch.from_numpy(batch_y).to(device).float()

                    log_prob = model(batch_x.float(), mode="log_prob", target_v=batch_y)
                    val_loss = -log_prob
                    val_loss_avg += val_loss.item() * batch_x.shape[0] / n_val

                    



                valid_loss.append(val_loss_avg)
                train_loss.append(loss_avg)
                if val_loss_avg < best_loss:
                    print('best so far (saving):')
                    best_loss = val_loss_avg
                    best_epoch = epoch
                    print("hi")
                    torch.save(model.state_dict(), save_folder+name+".pth")
                    print("bye")

                    print(
                        f'New best epoch {epoch}: train loss {loss_avg:.8f}, val loss {val_loss_avg:.8f}')

                else:
                    print(f'Epoch {epoch}: train loss {loss_avg:.8f}')
                if epoch - best_epoch > patience:
                    break

        else:
            print(f'Epoch {epoch}: train loss {loss_avg:.8f}')

    # model.load_state_dict(torch.load(save_folder+name+".pth"))
    # with torch.no_grad():
    #     model.eval()
    #     # X is a torch Variable
    #     n_val = x_test.shape[0]  # 1793
    #     permutation = torch.randperm(n_val)
    #
    #     val_loss_avg = 0
    #     for i in range(0, n_val, batch_size):
    #         indices = permutation[i:i + batch_size]  # size is 32 except the last batch (which may be < 32 )
    #
    #         batch_x, batch_y = x_test[indices], y_test[indices]
    #
    #         # convert to tensors
    #         batch_x = torch.from_numpy(batch_x).to(device)
    #         batch_y = torch.from_numpy(batch_y).to(device)
    #         n_batch = batch_x.shape[0]  # 32
    #         batch_y = batch_y.float()
    #
    #         # add augmentation later
    #
    #         outs = model(batch_x.float())  # (32, 10, 40)
    #
    #         # obtain the loss function
    #         # obtain the loss function
    #         loss = criterion(outs, batch_y)
    #
    #         val_loss_avg += loss.item() * n_batch / n_val
    # print(f"test loss: {val_loss_avg}")
    return train_loss,valid_loss






if __name__ == '__main__':



    # train
    train_loss, valid_loss = train_model("Weights/ETH/transformer/finetuning/","finetuning_eth_1", train_from_scratch =False)

