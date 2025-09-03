#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import peak_widths, medfilt

parser = argparse.ArgumentParser(description="Evaluate SNR from noise and denoised data")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

directory = Path(args.data_dir)
file_list = sorted(directory.glob(f'*{args.denoising_model}.h5'))


def get_snr(PSD, window=None):
    """Calculates SNR of PSD.

    Args:
        PSD: 1D array of PSD data.
        window: A tuple[peak, left, right] containing:
            peak: index where peak of data is.
            left: index where the left bin of the signal is.
            right: index where the right bin of the signal is.

            - default: None

    Returns:
        SNR: Signal-to-noise ratio of signal.
        window: same as above.
    """

    PSD = medfilt(PSD)  # applies median filter to data to remove any outliers

    # we calculate the window for the first pass of the function (noise PSD)
    if not window:      
        peak = np.argmax(PSD)
        _, _, left, right = peak_widths(PSD, [peak], rel_height=0.7, wlen=20001)
        window = (peak, int(left[0]), int(right[0]))
    
    peak       = window[0]
    left_edge  = window[1]
    right_edge = window[2]
    noise_bin  = 7500 #5000
    
    signal = np.mean(PSD[left_edge : right_edge + 1])

    noise  = np.concatenate([PSD[left_edge - noise_bin : left_edge - 200],
                             PSD[right_edge + 201      : right_edge + noise_bin + 1]])
    
    mean_noise = np.mean(noise)
    noise_std  = np.std(noise)

    snr = (signal - mean_noise) / noise_std
 
    return snr, window

snr   = []
snri  = []
scale = []

for i, filename in enumerate(file_list[:]):
    f = h5py.File(filename, 'r')
    print(filename.stem)

    # Data = log10(x + 1), so must do 10 ** (Data) - 1
    noise    = np.exp(f['input'][:]) - 1 
    denoised = np.exp(f['denoised'][:]) - 1
    injected = np.exp(f['injected'][:]) - 1

    scale_factor = (f['injected'].attrs['scale_factor'])
    scale.append(scale_factor)
    breakpoint()
    _, window = get_snr(injected)

    noise_snr    = get_snr(noise, window)
    denoised_snr = get_snr(denoised, window)

    snr.append(noise_snr[0])
    snri.append(denoised_snr[0] / noise_snr[0])

    plt.figure()
    plt.vlines([window[1], window[2]], 1e10, 1e12, lw=0.5, alpha=0.5, colors='r')
    # plt.plot([], label=f'Scale factor = {scale_factor:.3f}')
    plt.plot(noise, label=f'noise | snr = {noise_snr[0]:.3f}', c='b', alpha=0.5, lw=1)
    plt.plot(denoised, label=f'denoised | snr = {denoised_snr[0]:.3f}', c='r', alpha=0.5, lw=1)
    # plt.plot(injected, label=f'injected', alpha=0.5, lw=1)
    plt.yscale('log')
    plt.legend(loc='center left')
    plt.savefig(f'{args.data_dir}/{filename.stem}_FFT.pdf', dpi=300)
    plt.close()

# snr   = np.array(snr[3:])
snri  = np.array(snri[:])
#m, b = np.polyfit(snr, snri, 1)
scale = np.array(scale)

import numpy as np
from scipy.optimize import curve_fit

def negative_exponential(x, A, k, C):
    return A * np.exp(-k * x) + C

# Initial guesses for parameters (A, k, C)
p0 = [20, 2, 2] 
params, _ = curve_fit(negative_exponential, scale, snri, p0=p0)

# Extract the fitted parameters
A_fit, k_fit, C_fit = params

plt.figure()
plt.scatter(scale, snri, label='Data')
# plt.plot(snr, snr * m + b, label = f'LOB | Slope: {m:.2f}', c='red')
plt.plot(scale, negative_exponential(scale, A_fit, k_fit, C_fit), c='red', label=f'LOB')
plt.ylabel('SNRI')
plt.xlabel('Signal width')
plt.legend()
plt.savefig('SNRI(width)_plot.pdf', dpi=300)
breakpoint()
# plt.show()