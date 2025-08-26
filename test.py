#!/usr/bin/env python3

import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pyfftw

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory
parser.add_argument('--denoising_model', '-m', type=str, default='fcnet', help='Denoising model we would like to train [fcnet/punet/transformer/lwpt] (Default: punet).')
args = parser.parse_args()

def normalize_14bit(x):
    """Assumes input is a torch tensor in [0, 16383]"""
    return x / 16383.0

def denormalize_14bit(x):
    """Convert output back to 14-bit range"""
    return (x * 16383.0).clamp(0, 16383)

batchsize = 1
input_size = 40000

def digitize_data(data):
    "Takes in an array and digitizes it to 256 values"

    normalized = (data + 4) / (8)
    digitized = np.clip(normalized, 0, 1)
    digitized = np.round(normalized * 16383)
    
    return digitized

def read_loader(ADMXfile):
    data = ADMXfile['streams/stream0/acquisitions/0'][:]
    dims = np.shape(data)
    numrec = dims[0]
    numbuf = int(dims[1]/2)
    numpts = numrec*numbuf
    reshape_data = data[:].reshape(numpts,-1)
    cdata = pyfftw.empty_aligned(numpts, dtype=np.complex64)
    cdata = reshape_data[:,0] + 1j*reshape_data[:,1]
    raw_fft = pyfftw.interfaces.scipy_fftpack.fft(cdata)
    N_pos = len(raw_fft)
    if N_pos % 2 != 0:
        raw_fft = raw_fft[:-1]
        N_pos = N_pos - 1
    X_full = np.append(raw_fft,np.conj(raw_fft[1:-1][::-1]))
    ave_v_signal_reconstruct = pyfftw.interfaces.scipy_fftpack.ifft(X_full)
    ave_v_signal_reconstruct = np.append(ave_v_signal_reconstruct, [0, 0]).real
    print(pyfftw.interfaces.scipy_fftpack.rfft(ave_v_signal_reconstruct).argmax())
    train = digitize_data(ave_v_signal_reconstruct) - 8191
    print(pyfftw.interfaces.scipy_fftpack.rfft(train)[1:].argmax())
    max_index = int(4e7)
    train = train[:max_index].reshape(-1, batchsize, input_size)

    return train

directory = Path('D:\Run1D_admx')
file_list = list(directory.glob('*.egg'))

for fname in file_list:

    if args.denoising_model == 'fcnet':
        model = torch.load(f'FCNet_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == 'transformer':
        model = torch.load(f'Transformer_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == 'dcae':
        model = torch.load(f'DCAE_TS.pth', map_location=DEVICE, weights_only=False)
        model.eval()

    ADMXfile = h5py.File(fname,'r')
    noise = []
    denoised = []
    train_loader = read_loader(ADMXfile)
    # breakpoint()
    for i, batch in tqdm(enumerate(train_loader)):
        inputarr=  batch[:batchsize]
    
        input_seq = torch.from_numpy(inputarr)
        # Forward pass

        if args.denoising_model == 'dcae':
            input_seq = input_seq.float().to(DEVICE)
            output_seq = model(input_seq).detach().cpu().numpy()
        elif args.denoising_model == "transformer":
            input_seq = normalize_14bit(input_seq).float().unsqueeze(-1).to(DEVICE)
            # breakpoint()
            output_seq = 16383 * model(input_seq).detach().cpu().numpy()
        else:
            input_seq = input_seq.long().to(DEVICE)
            output_seq = model(input_seq).argmax(dim=1).detach().cpu().numpy()

        denoised.append(np.round(output_seq.flatten()))
        noise.append(inputarr.flatten())
        # breakpoint()

    denoised = np.concatenate(denoised, axis=0)
    noise = np.concatenate(noise, axis=0)
    print(denoised)
    # breakpoint()
    with h5py.File(f'{fname.stem}_denoised_{args.denoising_model}_TS.h5', 'w') as f:
        f.create_dataset('denoised', data=denoised)
        f.create_dataset('input', data=noise)


