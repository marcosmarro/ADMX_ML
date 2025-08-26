#!/usr/bin/env python3

import numpy as np
import os
import argparse
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
from statsmodels.graphics.tsaplots import plot_acf

parser = argparse.ArgumentParser(description="Evaluate SNR from noise and denoised data")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

directory = Path(args.data_dir)
file_list = sorted(directory.glob(f'*{args.denoising_model}.h5'))


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

snr  = []
snri = []

for i, filename in enumerate(file_list):
    f = h5py.File(filename, 'r')
    print(filename.stem)

    # Data = log10(x + 1), so must do 10 ** (Data) - 1
    noise    = np.exp(f['input'][:]) - 1 
    denoised = np.exp(f['denoised'][:]) - 1
    injected = np.exp(f['injected'][:]) - 1

    peak = injected.argmax()

    noise_snr, injected = get_snr(noise, peak)
    snr.append(noise_snr)
    denoised_snr = get_snr(denoised, peak)
    snri.append(denoised_snr[0] / noise_snr)
    print('Injected frequency:', peak)

    plt.figure()
    plt.plot(noise, label=f'noise | snr = {noise_snr:.3f}', c='b', alpha=0.5, lw=1)
    plt.plot(denoised, label=f'denoised | snr = {denoised_snr[0]:.3f}', c='r', alpha=0.5, lw=1)
    # plt.plot(injected, label=f'injected', alpha=0.5, lw=1)
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.savefig(f'{args.data_dir}/{filename.stem}_FFT.pdf', dpi=300)
    plt.close()

snr = np.array(snr)
snri = np.array(snri)

m, b = np.polyfit(snr, snri, 1)

plt.figure()
plt.scatter(snr, snri, label='Data')
plt.plot(snr, snr * m + b, c='red', label=f'LOB | Slope: {m:.2}')
plt.ylabel('SNRI')
plt.xlabel('SNR')
plt.legend()
plt.savefig('SNRI_plot.pdf', dpi=300)
breakpoint()