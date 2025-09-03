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
import pyfftw
from fft_network import AutoEncoder, DCAE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

# set the size of segmentations for deep learning models
sampling_freq = int(400e3)
input_size    = sampling_freq // 2 
batchsize     = 2


def read_loader(ADMXfile):
    train  = ADMXfile['input'][:]
    train  = train.reshape(-1, sampling_freq)
    train  = np.abs(pyfftw.interfaces.scipy_fft.rfft(train)[:, 1:]) ** 2
    train  = np.log1p(train)
    maximum = train.max()
    # train  = train / maximum
    train  = train.reshape(-1, batchsize, input_size)

    target = ADMXfile['injected'][:]
    target = target.reshape(-1, sampling_freq)
    target = np.abs(pyfftw.interfaces.scipy_fft.rfft(target)[:, 1:]) ** 2
    target = np.log1p(target) 
    # target = target / maximum
    target = target.reshape(-1, batchsize, input_size)
 
    return np.concatenate([train, target], axis=1)


if args.denoising_model == "fcnet":
    model     = AutoEncoder(input_size).to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)
elif args.denoising_model == "dcae":
    model     = DCAE().to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)
else:
    raise ValueError

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

directory = Path(args.data_dir)
file_list = sorted(directory.glob('*training.h5'))

for ii, fname in enumerate(file_list[:29]):
    print(fname.stem)
    ADMXfile     = h5py.File(fname, 'r')
    train_loader = read_loader(ADMXfile)
    np.random.shuffle(train_loader)

    for i, batch in enumerate(train_loader):
        inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
        input_seq  = torch.from_numpy(inputarr)
        target_seq = torch.from_numpy(targetarr)

        input_seq  = input_seq.float().to(DEVICE)
        target_seq = target_seq.float().to(DEVICE)

        # Forward pass
        output_seq = model(input_seq)
        loss       = criterion(output_seq, target_seq)

        # Backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 25 batches
        if i % 25 == 0:
            print(f'Epoch: {ii} | Batch: {i} | Loss: {loss.item()}')
            plt.plot(np.arange(input_size), input_seq[0].detach().cpu().numpy().flatten(), label = 'input',alpha=0.5, lw=1)
            plt.plot(np.arange(input_size), target_seq[0].detach().cpu().numpy().flatten(), label = 'target',alpha=0.5, lw=1)
            plt.plot(np.arange(input_size), output_seq[0].squeeze(-1).detach().cpu().numpy(), label = 'output',alpha=0.5, lw=1)
            plt.legend()
            plt.savefig(f"denoise_sample.pdf", dpi=100)
            plt.cla()
            plt.clf()
            plt.close()

    del ADMXfile, train_loader
    gc.collect()

if args.denoising_model == "fcnet":
    torch.save(model, 'FCNet_PSD.pth')

if args.denoising_model == "dcae":
    torch.save(model, 'DCAE_PSD.pth')

del model, criterion, optimizer
torch.cuda.empty_cache()