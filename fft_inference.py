#!/usr/bin/env python3

import numpy as np
import argparse
import torch
import h5py
from tqdm import tqdm
import os
from pathlib import Path
import pyfftw

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run PSD denoising model over science files.")
parser.add_argument('--data_dir', '-d', type=str, default=os.path.join(os.getcwd(),"Data"), help='Directory where the science files are stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

# set number of training batches and input size
sampling_freq = int(400e3)
input_size    = sampling_freq // 2
batchsize     = 1


def read_loader(ADMXfile):
    train = ADMXfile['input'][:]
    train = train.reshape(-1, sampling_freq)
    train = np.abs(pyfftw.interfaces.scipy_fft.rfft(train)[:, 1:]) ** 2
    train = np.log1p(train)
    maximum = train.max()
    # train = train / maximum
    train = train.reshape(-1, batchsize, input_size)
    
    target = ADMXfile['injected'][:]
    target = target.reshape(-1, sampling_freq)
    target = np.abs(pyfftw.interfaces.scipy_fft.rfft(target)[:, 1:]) ** 2
    target = np.log1p(target)
    # target = target / maximum
    target = target.reshape(-1, batchsize, input_size)

    return np.concatenate([train, target], axis=1), maximum


def main():

    if args.denoising_model == "dcae":
        model = torch.load(f'DCAE_PSD.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == "fcnet":
        model = torch.load(f'FCNet_PSD.pth', map_location=DEVICE, weights_only=False)
        model.eval() 			
    else:
        raise ValueError

    directory = Path(args.data_dir)
    file_list = sorted(directory.glob('*.h5'))

    for fname in tqdm(file_list[:]):
        ADMXfile     = h5py.File(fname,'r')
        train_loader, maximum = read_loader(ADMXfile)
        # breakpoint()
        noise    = []
        denoised = []
        injected = []
        
        for i, batch in enumerate(train_loader):
            inputarr, targetarr = (batch[:batchsize], batch[batchsize:])

            input_seq  = torch.from_numpy(inputarr)
            input_seq  = input_seq.float().to(DEVICE)

            output_seq = model(input_seq).detach().cpu().numpy()
            
            denoised.append(output_seq.flatten())
            injected.append(targetarr.flatten())
            noise.append(inputarr.flatten())

        denoised = np.mean(np.array(denoised), axis=0) 
        injected = np.mean(np.array(injected), axis=0) 
        noise    = np.mean(np.array(noise), axis=0) 
        
        print(denoised)

        with h5py.File(f'Denoised_fft_science/{fname.stem}_denoised_{args.denoising_model}.h5', 'w') as f:
            f.create_dataset('denoised', data=denoised)
            f.create_dataset('input', data=noise)
            f.create_dataset('injected', data=injected)

            f['injected'].attrs['sig_size'] = ADMXfile['injected'].attrs['sig_size']
            f['injected'].attrs['frequency_detune'] = ADMXfile['injected'].attrs['frequency_detune']
            f['injected'].attrs['scale_factor'] = ADMXfile['injected'].attrs['scale_factor']


if __name__ == "__main__":
	main()