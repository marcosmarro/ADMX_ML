#!/usr/bin/env python3

import numpy as np
import argparse
import torch
import h5py
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run time series denoising algorithm over the full validation dataset to produce denoised SQUID time series.")
parser.add_argument('--data_dir', '-d', type=str, default=os.path.join(os.getcwd(),"Data"), help='Directory where the science files are stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

#set number of training batches and input size
input_size = 40000
batchsize  = 20 


def read_loader(ADMXfile):
    train  = np.array(ADMXfile['input']) # - 8191
    target = np.array(ADMXfile['injected']) # - 8191

    max_index = int(4e7)
    train  = train[:max_index].reshape(-1, batchsize, input_size)
    target = target[:max_index].reshape(-1,  batchsize, input_size)

    return np.concatenate([train, target], axis=1)


def main():
    '''
    Reads in science files, loads selected model, and returns a denoised .h5 file with three channels:
        'input': the input signal for model to denoise (noise + injected)
        'injected': the injected/target signal 
        'denoised': the denoised signal
    '''

    if args.denoising_model == "dcae":
        model = torch.load(f'DCAE_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == "transformer":
        model = torch.load(f'Transformer_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == "fcnet":
        model = torch.load(f'FCNet_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()  	
    elif args.denoising_model == "unet":  
        model = torch.load(f'UNet_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()  
    else:
        raise ValueError

    directory = Path(args.data_dir)
    file_list = list(directory.glob('*.h5'))

    for fname in tqdm(file_list):
        ADMXfile     = h5py.File(fname,'r')
        train_loader = read_loader(ADMXfile)

        noise    = []
        denoised = []
        injected = []
        
        for i, batch in enumerate(train_loader):
            inputarr, targetarr = (batch[:batchsize], batch[batchsize:])

            input_seq  = torch.from_numpy(inputarr)
            input_seq  = input_seq.float().to(DEVICE)
            if args.denoising_model == 'unet':
                input_seq = input_seq.unsqueeze(1)

            output_seq = model(input_seq).detach().cpu().numpy()
        
            denoised.append(np.round(output_seq.flatten()))
            injected.append(targetarr.flatten())
            noise.append(inputarr.flatten())

        denoised = np.concatenate(denoised, axis=0)
        injected = np.concatenate(injected, axis=0)
        noise    = np.concatenate(noise, axis=0)
        print(denoised)

        with h5py.File(f'Denoised_science/{fname.stem}_denoised_{args.denoising_model}.h5', 'w') as f:
            f.create_dataset('denoised', data=denoised)
            f.create_dataset('input', data=noise)
            f.create_dataset('injected', data=injected)


if __name__ == "__main__":
	main()
