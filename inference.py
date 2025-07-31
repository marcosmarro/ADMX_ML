#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 2 2024
@author: TIDMAD Team
This code runs inference over the validation dataset to perform denoising task. Possible
denoising algorithms includes:

 - Moving Average [mavg]
 - Savitzky-Golay (SG) filter [savgol]
 - Fully Connected Network [fcnet]
 - Positional U-Net [punet]
 - Transformer [transformer]
"""

import numpy as np
import argparse
import torch
import h5py
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run time series denoising algorithm over the full validation dataset to produce denoised SQUID time series.")

# Output directory with default as current directory
parser.add_argument('--data_dir', '-d', type=str, default=os.path.join(os.getcwd(),"Data"), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [mavg/savgol/fcnet/punet/transformer] (Default: punet).')

args = parser.parse_args()


NON_ML = False
#set number of training batches and input ouput size
input_size = 40000
batchsize = 25 # input_size//batchsize is the length of each time series
if args.denoising_model == "savgol" or args.denoising_model == "mavg":
	input_size = 1000000
	batchsize = 1
	NON_ML = True
	window_size = 100

output_size = input_size
ADC_CHANNEL = 256



def read_loader(ADMXfile):
    train = np.array(ADMXfile['input'])
    target = np.array(ADMXfile['injected'])
    max_index = int(4e7)
    train = train[:max_index].reshape(-1, batchsize, input_size)
    target = target[:max_index].reshape(-1, batchsize, input_size)

    size = ADMXfile['injected'].attrs['log_sig_size']

    return np.concatenate([train, target], axis=1), size

def main():

    directory = Path(args.data_dir)
    file_list = list(directory.glob('*.h5'))
    for fname in file_list:

        if args.denoising_model == "punet":
            model = torch.load(f'PUNet.pth', map_location=DEVICE)
            model.eval()
        elif args.denoising_model == "transformer":
            model = torch.load(f'Transformer.pth', map_location=DEVICE)
            model.eval()
        elif args.denoising_model == "fcnet":
            model = torch.load(f'FCNet.pth', map_location=DEVICE, weights_only=False)
            model.eval()
        else:
            '''
            Do nothing, this is reserved for Non-ML models
            '''
            break

        ADMXfile = h5py.File(fname,'r')
        noise = []
        denoised = []
        injected = []
        train_loader, size = read_loader(ADMXfile)
        for i, batch in tqdm(enumerate(train_loader)):
            inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
            if args.denoising_model == "mavg":
                # Size of moving average kernel
                kernel = np.ones(window_size) / window_size
                output_seq = np.convolve(inputarr.flatten(), kernel,mode="same")
            elif args.denoising_model == "savgol":
                output_seq = savgol_filter(inputarr[0].flatten(), window_size, 11)
            elif args.denoising_model == "fcnet":
                input_seq = torch.from_numpy(inputarr)
                # Forward pass
                input_seq = input_seq.float().to(DEVICE)
                output_seq = model(input_seq).detach().cpu().numpy()
            else:
                input_seq = torch.from_numpy(inputarr)
                # Forward pass
                input_seq = input_seq.long().to(DEVICE)
                output_seq = model(input_seq).argmax(dim=1).detach().cpu().numpy()
            denoised.append(output_seq.flatten().astype(np.uint8))
            injected.append(targetarr.flatten())
            noise.append(inputarr.flatten())

        denoised = np.concatenate(denoised, axis=0)
        injected = np.concatenate(injected, axis=0)
        noise = np.concatenate(noise, axis=0)
        print(denoised)
        # breakpoint()
        with h5py.File(f'{fname.stem}_denoised.h5', 'w') as f:
            f.create_dataset('denoised', data=denoised)
            f.create_dataset('input', data=noise)
            f.create_dataset('injected', data=injected)
            f['injected'].attrs['log_sig_size'] = size

if __name__ == "__main__":
	main()
