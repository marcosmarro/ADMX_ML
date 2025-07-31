#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 2024
@author: TIDMAD Team

This script trains deep learning model over the training dataset, possible architecture includes:
 - Fully Connected Network [fcnet]
 - Positional U-Net [punet]
 - Transformer [transformer]
"""

import numpy as np
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import gc
import os
import lwpt
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from network import PositionalUNet, FocalLoss1D, TransformerModel, AE

# SQUID = h5py.File(SQUIDname,'r')
# SG = h5py.File(SGname, 'r')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='fcnet', help='Denoising model we would like to train [fcnet/punet/transformer/lwpt] (Default: punet).')

args = parser.parse_args()

#set the size of segmentations for deep learning models
input_size = 40000
if args.denoising_model == "transformer":
	input_size = 20000 # transformer model requires additional GPU memories, so we reduce segment size by 50%
sample_size = 2 #Randomly sample 50% of the time series to train model
batchsize = 1
output_size = input_size


def read_loader(ADMXfile):
    train = np.array(ADMXfile['input']) - 1023
    target = np.array(ADMXfile['injected']) - 1023
    # breakpoint()
    max_index = int(4e7)
    train = train[:max_index].reshape(-1, sample_size, batchsize, input_size)
    target = target[:max_index].reshape(-1, sample_size, batchsize, input_size)
    random_index = np.random.randint(sample_size)

    return np.concatenate([train[:, random_index], target[:, random_index]], axis=1)


if args.denoising_model == "punet":
    model = PositionalUNet().to(DEVICE)
    criterion = FocalLoss1D().to(DEVICE)
elif args.denoising_model == "transformer":
    model = TransformerModel().to(DEVICE)
    criterion = FocalLoss1D().to(DEVICE)
elif args.denoising_model == "fcnet":
    model = AE(input_size).to(DEVICE)
    criterion = nn.SmoothL1Loss().to(DEVICE)
elif args.denoising_model == 'lwpt':
    model = lwpt.NeuralDWAV(input_size,
                              Input_Level=5,#WPT with 5 level resulting to 32 outputs
                              Input_Archi="WPT").to(DEVICE)
    criterion = torch.nn.MSELoss()
else:
    raise ValueError

directory = Path(args.data_dir)
file_list = list(directory.glob('*training.h5'))

for ii, fname in enumerate(file_list):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Read file
    ADMXfile = h5py.File(fname, 'r')
    # print(ADMXfile['injected'].attrs['frequency'])
    # Start training
    train_loader = read_loader(ADMXfile)
    np.random.shuffle(train_loader)
    for i, batch in enumerate(train_loader):
        inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
        input_seq = torch.from_numpy(inputarr)
        target_seq = torch.from_numpy(targetarr)
        randind = np.random.randint(batchsize)
        input_seq = input_seq[randind].unsqueeze(0).float().to(DEVICE)
        target_seq = target_seq[randind].unsqueeze(0).float().to(DEVICE)
        # Forward pass
        if args.denoising_model == "punet" or args.denoising_model == "transformer":
            input_seq = input_seq.int()
            target_seq = target_seq.long()
        if args.denoising_model == "lwpt":
            input_seq = input_seq.unsqueeze(0).double()
            target_seq = target_seq.unsqueeze(0).double()
        # breakpoint()
        output_seq = model(input_seq)

        # Calculate the loss
        loss = criterion(output_seq, target_seq)

        # Backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()


        optimizer.step()

        # Print the loss every 100 batches
        if (i % 100) == 0:
            print('Epoch: {} | Batch: {} | Loss: {}'.format(ii, i, loss.item()))
            plt.plot(np.arange(input_size), input_seq[0].detach().cpu().numpy().flatten(), label = 'input',alpha=0.5, lw=1)
            plt.plot(np.arange(input_size), target_seq[0].detach().cpu().numpy().flatten(), label = 'target',alpha=0.5, lw=1)
            if args.denoising_model == "punet" or args.denoising_model == "transformer":
                # If the model is not FCNet, the model accomplish a segmentation task with 256 classes per time step
                output_seq = output_seq.argmax(dim=1)
            plt.plot(np.arange(input_size), output_seq[0].detach().cpu().numpy().flatten(), label = 'output',alpha=0.5, lw=1)
            plt.legend()
            plt.savefig("denoise_sample.pdf",dpi=100)
            plt.cla()
            plt.clf()
            plt.close()

    del ADMXfile, train_loader
    gc.collect()


if args.denoising_model == "fcnet":
    torch.save(model, 'FCNet.pth')

if args.denoising_model == "punet":
    torch.save(model, 'PUNet.pth')

if args.denoising_model == "transformer":
    torch.save(model, 'Transformer.pth')

if args.denoising_model == "lwpt":
    torch.save(model, 'LWPT.pth')

del model, criterion, optimizer
torch.cuda.empty_cache()