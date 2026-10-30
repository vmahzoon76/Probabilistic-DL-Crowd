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
    """Your original Transformer-based velocity predictor."""
    def __init__(self, inputs, pred_type, out_dim, n_objects, nhead, nhid, nlayers, dropout=0.0):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.nhid = nhid
        self.inputs = inputs
        self.out_dim = out_dim
        # initial MLP
        self.fc1 = nn.Sequential(
            nn.Linear(inputs, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
        )
        # transformer stack
        enc_layer = TransformerEncoderLayer(d_model=nhid, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(enc_layer, num_layers=nlayers)
        # output MLP
        self.out = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, out_dim),
        )

    def forward(self, src):
        # src: [B, P, E]
        P, B, E = src.shape[1], src.shape[0], src.shape[2]
        # permute to [P, B, E]
        h = src.permute(1, 0, 2)
        h = self.fc1(h)                      # [P, B, nhid]
        h = self.transformer_encoder(h)      # [P, B, nhid]
        h = h.transpose(0, 1)                # [B, P, nhid]
        # select only center (first) agent at last time
        out = h[:, 0:1, :]         # [B, 1, out_dim]
        return out


class DiffusionVelocityPredictor(nn.Module):
    """Diffusion wrapper around TransformerModel to predict noise epsilon."""
    def __init__(self, transformer: TransformerModel, T: int = 200):
        super().__init__()
        self.context_net = transformer
        self.hidden = transformer.nhid
        self.T = T
        # diffusion schedule buffers
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_bar', alphas_bar)
        # time embedding
        self.time_emb = nn.Embedding(T, self.hidden)
        # epsilon predictor head
        self.eps_head = nn.Sequential(
            nn.Linear(self.hidden * 2 + 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2)
        )

    def q_sample(self, v0, t):
        """Add noise to clean velocity v0 at timestep t."""
        noise = torch.randn_like(v0)
        a_bar = self.alphas_bar[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * v0 + torch.sqrt(1 - a_bar) * noise, noise

    def forward(self, src, v_t, t):
        # src: [B, P, E], v_t: [B, 1, 2], t: [B]
        B = src.size(0)
        # get context embedding [B, hidden]
        h = self.context_net(src).view(B, -1)
        # embed t [B, hidden]
        t_e = self.time_emb(t)
        # flatten v_t [B,2]
        v_flat = v_t.view(B, 2)
        # combine [h, v, t_e]
        inp = torch.cat([h, v_flat, t_e], dim=-1)
        # predict noise [B, 2]
        eps = self.eps_head(inp)
        return eps


def sample_velocity(model, src, T=None):
    # src: [B, P, E]
    if T is None:
        T = model.T

    B = src.shape[0]
    temp = np.random.uniform(0.2, 1.4)

    # scale initial noise — this controls rollout diversity 🔥
    v_t = torch.randn(B, 1, 2, device=src.device) * temp
   

    for t in reversed(range(T)):
        t_idx = torch.full((B,), t, dtype=torch.long, device=src.device)
        eps_pred = model(src, v_t, t_idx)  # [B, 2]

        beta_t = model.betas[t]
        alpha_t = 1.0 - beta_t
        a_bar_t = model.alphas_bar[t]
        a_bar_prev = model.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=src.device)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            v_t - (beta_t / torch.sqrt(1.0 - a_bar_t)) * eps_pred.view(B, 1, 2)
        )

        if t > 0:
            sigma_t = torch.sqrt((1.0 - a_bar_prev) / (1.0 - a_bar_t) * beta_t)
            mean += sigma_t * torch.randn_like(v_t)

        v_t = mean

    return v_t.squeeze(1).detach().cpu().numpy()  # [B, 2]


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





def train_model(save_folder,name,train_type='early_stopping', train_from_scratch = True):
    train_loss=[]
    valid_loss=[]
    start_time = datetime.now()
    device = 'cuda:0'
    dt = 0.10


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

    x_test,y_test=get_data("Datasets/Hotel/data_for_model/transformer/Hotel_train_part_2.pkl",27)

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
    # build diffusion model
    base = TransformerModel(inputs=6, pred_type="trajectory", out_dim=2,
                            n_objects=27, nhead=8, nhid=128, nlayers=4).to(device)
    

    model = DiffusionVelocityPredictor(base, T=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not train_from_scratch:
        ckpt = torch.load("Weights/pretrained/transformer/god.pth", map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)


    # lin_train_loss = criterion(torch.from_numpy(x_train[:, 0:1, 2:4].copy()),
    #                            torch.from_numpy(y_train)).detach().numpy()
    # print("MSE of velocity")
    # print(f'train loss {lin_train_loss:.8f}')

    # lin_val_loss = criterion(torch.from_numpy(x_val[:, 0:1, 2:4].copy()), torch.from_numpy(y_val)).detach().numpy()
    # print(f'validation loss {lin_val_loss:.8f}')
    # print()

    # lin_val_loss = criterion(torch.from_numpy(x_test[:, 0:1, 2:4].copy()), torch.from_numpy(y_test)).detach().numpy()
    # print(f'test loss {lin_val_loss:.8f}')
    # print()

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

            # sample t and noise
            B = batch_x.size(0)
            t = torch.randint(0, model.T, (B,), device=device)
            v_t, noise = model.q_sample(batch_y, t)
            pred = model(batch_x.float(), v_t, t)
            loss = F.mse_loss(pred, noise.view(B, 2))


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

                    B = batch_x.size(0)
                    t = torch.randint(0, model.T, (B,), device=device)
                    v_t, noise = model.q_sample(batch_y, t)
                    pred = model(batch_x.float(), v_t, t)
                    loss = F.mse_loss(pred, noise.view(B, 2))


                    val_loss_avg += loss.item() * n_batch / n_val
                valid_loss.append(val_loss_avg)
                train_loss.append(loss_avg)
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

   
    
    return train_loss,valid_loss






if __name__ == '__main__':
    # train
    train_loss, valid_loss = train_model("Weights/ETH/transformer/finetuning/","finetuning_eth_1", train_from_scratch =False)
