import numpy as np
import os
import argparse
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import pyfftw


parser = argparse.ArgumentParser(description="Evaluate SNR from noise and denoised data")

parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')

args = parser.parse_args()

directory = Path(args.data_dir)
file_list = list(directory.glob('*transformer.h5'))

def get_snr(fft, injected_freq = 0):

    if injected_freq == 0:
        injected_freq = np.argmax(fft)
    signal = np.sum(fft[injected_freq - 10: injected_freq + 11])
    noise = np.sum(fft[injected_freq - 50: injected_freq + 51]) - signal
    snr = np.average(signal) / np.average(noise)

    return snr, injected_freq

for filename in file_list:
    f = h5py.File(filename, 'r')
    noise = f['input'][:]
    denoised = f['denoised'][:]
    # injected = f['injected'][:]
    # breakpoint()
    # size = np.round(f['injected'].attrs['log_sig_size'], decimals=2)

    noise_fft = pyfftw.interfaces.scipy_fft.rfft(noise).real[1:] ** 2
    denoised_fft = pyfftw.interfaces.scipy_fft.rfft(denoised).real[1:] ** 2
    # injected_fft = pyfftw.interfaces.scipy_fft.rfft(injected).real[1:] ** 2
    chunk = int(1e2)
    binned_noise_fft = np.empty(int(len(noise_fft) / chunk))
    binned_denoised_fft = np.empty(int(len(noise_fft) / chunk))
    # binned_injected_fft = np.empty(int(len(noise_fft) / chunk))

    for i in range(0, len(binned_noise_fft)):
        binned_noise_fft[i] = np.average(noise_fft[chunk*i : chunk*i+chunk])
        binned_denoised_fft[i] = np.average(denoised_fft[chunk*i : chunk*i+chunk])
        # binned_injected_fft[i] = np.average(injected_fft[chunk*i : chunk*i+chunk])

    noise_snr, injected = get_snr(binned_noise_fft)
    denoised_snr = get_snr(binned_denoised_fft, injected)
    # breakpoint()
    # plt.figure()
    # # plt.plot(injected, lw=0.5, label='injected')
    # plt.plot(noise[:int(2e6)], lw=0.5, label='input', alpha=0.5)
    # plt.plot(denoised[:int(2e6)], lw=0.5, label='denoised', alpha=0.5)
    # # plt.plot(injected[:int(2e6)], lw=0.5, label='injected', alpha=0.5)
    # plt.legend()
    # plt.savefig(f'{filename.stem}_TS.pdf', dpi=300)
    # # plt.show()
    # plt.close()

    plt.figure()
    plt.plot(binned_noise_fft, label=f'noise | snr = {noise_snr:3f}', c='b', alpha=0.5, lw=1)
    plt.plot(binned_denoised_fft, label=f'denoised | snr = {denoised_snr[0]:3f}', c='r', alpha=0.5, lw=1)
    # plt.plot(binned_injected_fft, label=f'injected', c='orange', alpha=0.5, lw=1)
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'both_11bit_{filename.stem}_FFT.pdf', dpi=300)
    plt.close()
