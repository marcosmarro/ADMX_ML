#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import gc
import os
from network import FocalLoss1D, TransformerModel, AutoEncoder, DCAE, UNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/transformer/dcae] (Default: dcae).')
args = parser.parse_args()

#set the size of segmentations for deep learning models
input_size  = 40000
sample_size = 2 # Randomly sample 25% of the time series to train model (1 / sample_size)
batchsize   = 1
if args.denoising_model == "transformer":
	input_size = 20000 # transformer model requires additional GPU memories, so we reduce segment size by 50%


def read_loader(ADMXfile):
    train = np.array(ADMXfile['input'])  #- 8191
    target = np.array(ADMXfile['injected'])  #- 8191
    # breakpoint()
    max_index = int(4e7)
    train = train[:max_index].reshape(-1, sample_size, batchsize, input_size)
    target = target[:max_index].reshape(-1, sample_size, batchsize, input_size)
    random_index = np.random.randint(sample_size)

    return np.concatenate([train[:, random_index], target[:, random_index]], axis=1)


if args.denoising_model == "transformer":
    model     = TransformerModel().to(DEVICE)
    criterion = FocalLoss1D().to(DEVICE)
elif args.denoising_model == "fcnet":
    model     = AutoEncoder(input_size).to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)
elif args.denoising_model == "dcae":
    model     = DCAE().to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)
elif args.denoising_model == "unet":
    model     = UNet(depth=2).to(DEVICE)
    criterion = nn.SmoothL1Loss().to(DEVICE)
else:
    raise ValueError

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

directory = Path(args.data_dir)
file_list = list(directory.glob('*training.h5'))


for ii, fname in enumerate(file_list):
    ADMXfile     = h5py.File(fname, 'r')
    train_loader = read_loader(ADMXfile)
    np.random.shuffle(train_loader)

    for i, batch in enumerate(train_loader):
        inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
        input_seq  = torch.from_numpy(inputarr)
        target_seq = torch.from_numpy(targetarr)

        randint    = np.random.randint(batchsize)
        input_seq  = input_seq[randint].unsqueeze(0).float().to(DEVICE)
        input_seq = input_seq.unsqueeze(1)
        target_seq = target_seq[randint].unsqueeze(0).float().to(DEVICE)
        target_seq = target_seq.unsqueeze(1)
        # breakpoint()
        # Forward pass
        output_seq = model(input_seq)
        loss       = criterion(output_seq, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 batches
        if (i % 100) == 0:
            print('Epoch: {} | Batch: {} | Loss: {}'.format(ii, i, loss.item()))
            plt.plot(np.arange(input_size), input_seq[0].detach().cpu().numpy().flatten(), label = 'input',alpha=0.5, lw=1)
            plt.plot(np.arange(input_size), target_seq[0].detach().cpu().numpy().flatten(), label = 'target',alpha=0.5, lw=1)
            plt.plot(np.arange(input_size), output_seq[0, 0].squeeze(-1).detach().cpu().numpy(), label = 'output',alpha=0.5, lw=1)
            plt.legend()
            plt.savefig("denoise_sample.pdf",dpi=100)
            plt.cla()
            plt.clf()
            plt.close()

    del ADMXfile, train_loader
    gc.collect()

if args.denoising_model == "fcnet":
    torch.save(model, 'FCNet_TS.pth')

if args.denoising_model == "transformer":
    torch.save(model, 'Transformer_TS.pth')

if args.denoising_model == "dcae":
    torch.save(model, 'DCAE_TS.pth')

if args.denoising_model == "unet":
    torch.save(model, 'Unet_TS.pth')

del model, criterion, optimizer
torch.cuda.empty_cache()