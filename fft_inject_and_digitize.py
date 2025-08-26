#!/usr/bin/env python3

import h5py
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from frequency_simulation import make_injected
import pyfftw
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 1000

parser = argparse.ArgumentParser(description="Digitizes the data and injects a classical random field")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Data directory where data files are stored')
parser.add_argument('--type', '-t', type=str, default='Training', help='Type of file to inject')
args = parser.parse_args()


def digitize_data(data):
    "Takes in an array and digitizes it to 16384 values (14-bit)"

    normalized = (data + 4) / (8)
    digitized = np.clip(normalized, 0, 1)
    digitized = np.round(normalized * 16383)
    
    return digitized

def exponential_psd(fft):
    freq_bins = fft.shape[0]
    psd = 2e11 * np.random.exponential(1, size=freq_bins) + fft

    return psd

def main():
    '''
    Runs the script in the following order:
        1. Reads the pure noise .h5 files 
        2. Generates an injected signal
        3. Digitizes both the injected signal and noise+injected
        3. Writes a .h5 file with two channels:
            'input': the input signal for training (noise + injected)
            'injected': the injected/target signal for training 
    '''

    sampling_freq = int(400e3)
    input_size    = sampling_freq * 100 # sampling frequency * 100 seconds

    if args.type == 'Training':
        np.random.seed(12) 
    elif args.type == 'Science':
        np.random.seed(56)
    sig_size = 10 ** 5
    detune = 0
    breakpoint()
    for i in range(40):
        raw_injected = make_injected(input_size, sig_size, detune)
        injected_data = digitize_data(raw_injected)
        injected_psd = np.abs(pyfftw.interfaces.scipy_fft.rfft(injected_data).real[1:]) ** 2

        input_psd = exponential_psd(injected_psd)

        plt.plot(np.log1p(injected_psd), alpha=0.5, label='injected')
        plt.plot(np.log1p(input_psd), alpha=0.5, label='noise')
        plt.legend()
        # plt.yscale('log')
        plt.show()
        breakpoint()
       
        f = h5py.File(f'8bit_Science/{filename.stem}_science.h5', 'w') # change to training/science for different needs
        f.create_dataset('input', data=input_data)
        f.create_dataset('injected', data=injected_data)

        f['injected'].attrs['sig_size'] = sig_size[i]
        f['injected'].attrs['frequency_detune'] = detune[i]
        

if __name__ == '__main__':
    main()