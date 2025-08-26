#!/usr/bin/env python3

import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse
import pyfftw
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

sampling_freq = int(400e3)
input_size    = sampling_freq // 2 
batchsize     = 1


def digitize_data(data):
    "Takes in an array and digitizes it to 16384 values (14-bit)"

    normalized = (data + 4) / (8)
    digitized = np.clip(normalized, 0, 1)
    digitized = np.round(normalized * 16383)
    
    return digitized


def read_loader(ADMXfile):
    # data = ADMXfile['streams/stream0/acquisitions/0'][:]
    # dims = np.shape(data)
    # numrec = dims[0]
    # numbuf = int(dims[1]/2)
    # numpts = numrec*numbuf
    # reshape_data = data[:].reshape(numpts,-1)
    # cdata = pyfftw.empty_aligned(numpts, dtype=np.complex64)
    # cdata = reshape_data[:,0] + 1j*reshape_data[:,1]
    # raw_fft = pyfftw.interfaces.scipy_fftpack.fft(cdata)
    # N_pos = len(raw_fft)
    # if N_pos % 2 != 0:
    #     raw_fft = raw_fft[:-1]
    #     N_pos = N_pos - 1
    # X_full = np.append(raw_fft,np.conj(raw_fft[1:-1][::-1]))

    # ave_v_signal_reconstruct = pyfftw.interfaces.scipy_fftpack.ifft(X_full)
    # ave_v_signal_reconstruct = np.append(ave_v_signal_reconstruct, [0, 0]).real

    # train = digitize_data(ave_v_signal_reconstruct)

    train = np.append(ADMXfile['Data'][:].real, [0,0])
    train = digitize_data(train)
    train = train.reshape(-1, sampling_freq)
    train = np.abs(pyfftw.interfaces.scipy_fft.rfft(train)[:, 1:]) ** 2
    train = np.log1p(train)
    maximum = train.max()
    #train = train / maximum
    train = train.reshape(-1, batchsize, input_size)
    # breakpoint()
    return train, maximum


if args.denoising_model == 'fcnet':
    model = torch.load(f'FCNet_PSD.pth', map_location=DEVICE, weights_only=False)
    model.eval()
elif args.denoising_model == 'dcae':
    model = torch.load(f'DCAE_PSD.pth', map_location=DEVICE, weights_only=False)
    model.eval()
else:
    raise ValueError

# directory = Path('D:/Run1D_admx')
# file_list = list(directory.glob('*.egg'))

directory = Path('./')
file_list = list(directory.glob('uninjected*.h5'))

for fname in file_list:
    ADMXfile     = h5py.File(fname,'r')
    if not ADMXfile['Data']:
        continue
    train_loader, maximum = read_loader(ADMXfile)

    noise = []
    denoised = []

    for i, batch in tqdm(enumerate(train_loader)):
        inputarr  = batch[:batchsize]

        input_seq = torch.from_numpy(inputarr)
        input_seq = input_seq.float().to(DEVICE)

        output_seq = model(input_seq).detach().cpu().numpy()

        denoised.append(output_seq.flatten())
        noise.append(inputarr.flatten())

    denoised = np.mean(np.array(denoised), axis=0)  #* maximum
    noise    = np.mean(np.array(noise), axis=0)  #* maximum
    print(denoised)

    with h5py.File(f'{fname.stem}_denoised_{args.denoising_model}.h5', 'w') as f:
        f.create_dataset('denoised', data=denoised)
        f.create_dataset('input', data=noise)