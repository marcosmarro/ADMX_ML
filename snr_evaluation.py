#!/usr/bin/env python3

import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import peak_widths, medfilt

parser = argparse.ArgumentParser(description="Evaluate SNR from noise and denoised data")
parser.add_argument('--data_dir', '-d', type=str, default='Denoised_science/', help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()


def get_window(PSD: np.ndarray) -> tuple:
    """Calculates window to calculate SNR for signal.

    Args:
        PSD: 1D array of injected PSD data.

    Returns:
        window: A tuple[peak, left, right] containing:
            left: index where the left bin of the signal is.
            right: index where the right bin of the signal is.
    """
    PSD = medfilt(PSD)

    # We calculate the window for the injected PSD
    peak = np.argmax(PSD)
    _, _, left, right = peak_widths(PSD, [peak], rel_height=0.7, wlen=20001)
    window = (int(left[0]), int(right[0]))

    return window


def get_snr(PSD: np.ndarray, window:tuple) -> float:
    """Calculates SNR of PSD.

    Args:
        PSD: 1D array of PSD data.
        window: A tuple[peak, left, right] containing:
            left: index where the left bin of the signal is.
            right: index where the right bin of the signal is.

    Returns:
        SNR: Signal-to-noise ratio of signal.
    """
    left_edge  = window[0]
    right_edge = window[1]
    noise_bin  = 10000 #5000
    
    signal = np.mean(PSD[left_edge : right_edge + 1])

    noise  = np.concatenate([PSD[left_edge - noise_bin : left_edge - 200],
                             PSD[right_edge + 201      : right_edge + noise_bin + 1]])
    
    mean_noise = np.mean(noise)
    noise_std  = np.std(noise)

    snr = (signal - mean_noise) / noise_std
    sig.append(signal - mean_noise)

    return snr


directory = Path(args.data_dir)
file_list = sorted(directory.glob(f'*{args.denoising_model}.h5'))

snr   = []
snri  = []
scale = []
sig   = []

for i, filename in enumerate(file_list[:]):
    f = h5py.File(filename, 'r')

    # Data = log(x + 1), so must do e^(Data) - 1
    noise    = np.exp(f['input'][:]) - 1 
    denoised = np.exp(f['denoised'][:]) - 1
    injected = np.exp(f['injected'][:]) - 1

    scale_factor = (f['injected'].attrs['scale_factor'])
    scale.append(scale_factor)
    
    window = get_window(injected)

    noise_snr    = get_snr(noise, window)
    denoised_snr = get_snr(denoised, window)
    
    snr.append(noise_snr)
    snri.append(denoised_snr / noise_snr)

    plt.figure()
    plt.vlines([window[0], window[1]], 1e10, 1e12, lw=0.5, alpha=0.5, colors='r')
    plt.plot(noise, label=f'noise | snr = {noise_snr:.3f}', c='b', alpha=0.5, lw=1)
    plt.plot(denoised, label=f'denoised | snr = {denoised_snr:.3f}', c='r', alpha=0.5, lw=1)
    plt.yscale('log')
    plt.legend(loc='center left')
    plt.savefig(f'{args.data_dir}/{filename.stem}_FFT.pdf', dpi=300)
    plt.close()

    print(f'Evaluated {filename.stem}')