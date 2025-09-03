#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from frequency_simulation import make_injected

mpl.rcParams['agg.path.chunksize'] = 1000

parser = argparse.ArgumentParser(description="Digitizes the data and injects an axion signal")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Data directory where data files are stored')
parser.add_argument('--type', '-t', type=str, default='Training', help='Type of file to inject [Training/Science]')
args = parser.parse_args()


def digitize_data(data):
    "Takes in an array and digitizes it to 16384 values (14-bit)"

    normalized = (data + 4) / (8)
    digitized = np.clip(normalized, 0, 1)
    digitized = np.round(normalized * 16383)
    
    return digitized


def main():
    """
    Runs the script in the following order:
        1. Reads the pure noise .h5 files 
        2. Generates an injected signal
        3. Digitizes both the injected signal and noise+injected
        3. Writes a .h5 file with two channels:
            'input': the input signal for training (noise + injected)
            'injected': the injected/target signal for training 
    """

    directory = Path(f'D:/Run1D_admx/corrected_data/{args.type}') # Training/Science
    files = sorted(directory.glob('*.h5'))

    # Sets seed number for different type of file
    if args.type == 'Training':
        np.random.seed(11) 
    elif args.type == 'Science':
        np.random.seed(55)

    # Signal size of axion and how offset the frequency is from the resonant frequency
    detune   = np.round(np.random.triangular(-9, 0, 9, size=len(files)), decimals=3) * 1e4
    detune   = 0
    # sig_size = np.linspace(0, 1e3, num=len(files), dtype=int) # 6.3 max + 
    # sig_size = [1.5e2, 1.5e3, 1.5e4, 1.5e5, 1.5e6]
    sig_size = 1.5e3
    scale    = np.linspace(.5, 3, len(files))


    for i, filename in tqdm(enumerate(files[:])):
        raw_noise    = h5py.File(filename, 'r')['Data'][:].real
        raw_noise    = np.append(raw_noise, [0, 0])
        raw_injected = make_injected(len(raw_noise), sig_size, detune, scale_factor=scale[i])
 
        input_data    = digitize_data(raw_noise + raw_injected) 
        injected_data = digitize_data(raw_injected) 
        # # breakpoint()
        # import pyfftw
        # fs = int(400e3)
        # input_data = input_data.reshape(-1, fs)
        # injected_data = injected_data.reshape(-1, fs)
        # injected_fft = np.abs(pyfftw.interfaces.scipy_fft.rfft(injected_data)[:, 1:]) **2 
        # noise_fft = np.abs(pyfftw.interfaces.scipy_fft.rfft(input_data)[:, 1:]) **2
        # plt.plot(np.log1p(injected_fft.mean(axis=0)), alpha=0.5, label='injected')
        # plt.plot(np.log1p(noise_fft.mean(axis=0)), alpha=0.5, label='noise')
        # # plt.plot(raw_injected)
        # # plt.plot(pyfftw.interfaces.scipy_fft.rfft(raw_noise + raw_injected)[1:].real ** 2)
        # plt.legend(loc='lower left')
        # # plt.yscale('log')
        # plt.show()
        # breakpoint()
       
        f = h5py.File(f'{args.type}/{filename.stem}_{args.type.lower()}.h5', 'w') # change to training/science for different needs
        f.create_dataset('input', data=input_data)
        f.create_dataset('injected', data=injected_data)

        f['injected'].attrs['sig_size'] = sig_size
        f['injected'].attrs['frequency_detune'] = detune
        f['injected'].attrs['scale_factor'] = scale[i]
        print(f'{filename.name} injected')
        

if __name__ == '__main__':
    main()