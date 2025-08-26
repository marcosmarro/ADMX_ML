#!/usr/bin/env python3

import numpy as np
import os
import argparse
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import pyfftw

parser = argparse.ArgumentParser(description="Evaluate SNR from noise and denoised data")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae/transformer] (Default: dcae).')
args = parser.parse_args()

directory = Path(args.data_dir)
file_list = list(directory.glob(f'*{args.denoising_model}.h5'))



def get_snr(fft, peak = 0):
    if peak == 0:
        peak = np.argmax(fft)

    left_sig_bin  = 350  # 600 14-bit, 200 tone injection
    right_sig_bin = 1000
    noise_bin     = 5000 
    
    signal = np.mean(fft[peak - left_sig_bin : peak + right_sig_bin + 1])

    noise  = np.concatenate([fft[peak - noise_bin   : peak - left_sig_bin - 500],
                             fft[peak + right_sig_bin + 501 : peak + noise_bin + 1]])
    
    mean_noise = np.mean(noise)
    noise_std  = np.std(noise)
    # breakpoint()
    snr = (signal - mean_noise) / noise_std
   
    return snr, peak


for filename in file_list:
    f = h5py.File(filename, 'r')

    fs = int(400e3)    
    noise    = f['input'][:].reshape(-1, fs)
    denoised = f['denoised'][:].reshape(-1, fs)
    injected = f['injected'][:].reshape(-1, fs)

    noise_fft    = np.abs(pyfftw.interfaces.scipy_fft.rfft(noise)[:, 1:]) ** 2
    denoised_fft = np.abs(pyfftw.interfaces.scipy_fft.rfft(denoised)[:, 1:]) ** 2
    injected_fft = np.abs(pyfftw.interfaces.scipy_fft.rfft(injected)[:, 1:]) ** 2

    noise_fft = np.mean(noise_fft, axis=0)
    denoised_fft = np.mean(denoised_fft, axis=0)
    injected_fft = np.mean(injected_fft, axis=0)
    # Bins fft into chunks 
    # chunk = int(1e3)
    # binned_noise_fft = np.empty(int(len(noise_fft) / chunk))
    # binned_denoised_fft = np.empty(int(len(noise_fft) / chunk))
    # binned_injected_fft = np.empty(int(len(noise_fft) / chunk))
    # for i in range(0, len(binned_noise_fft)):
    #     binned_noise_fft[i] = np.average(noise_fft[chunk*i : chunk*i+chunk])
    #     binned_denoised_fft[i] = np.average(denoised_fft[chunk*i : chunk*i+chunk])
    #     binned_injected_fft[i] = np.average(injected_fft[chunk*i : chunk*i+chunk])

    # Calculate SNR for the input and denoised 
    peak = injected_fft.argmax()
    noise_snr, injected_sig = get_snr(noise_fft, peak)
    denoised_snr = get_snr(denoised_fft, peak)
    print('Injected frequency:', peak)

    plt.figure()
    plt.plot(noise_fft, label=f'noise | snr = {noise_snr:.3f}', c='b', alpha=0.5, lw=1)
    plt.plot(denoised_fft, label=f'denoised | snr = {denoised_snr[0]:.3f}', c='r', alpha=0.5, lw=1)
    # plt.plot(injected_fft, label=f'injected', alpha=0.5, lw=1)
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(f'{args.data_dir}/{filename.stem}_FFT.pdf', dpi=300)
    plt.close()