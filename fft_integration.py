#!/usr/bin/env python3

import os
import h5py
import torch
import pyfftw
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.signal import peak_widths, medfilt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run PSD denoising model over science files.")
parser.add_argument('--data_dir', '-d', type=str, default=os.path.join(os.getcwd(),"Data"), help='Directory where the science files are stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='dcae', help='Denoising model we would like to train [fcnet/dcae] (Default: dcae).')
args = parser.parse_args()

# set number of training batches and input size
sampling_freq = int(400e3)
input_size    = sampling_freq // 2
batchsize     = 1


def get_snr(PSD, window=None):
    """Calculates SNR of PSD.

    Args:
        PSD: 1D array of PSD data
        window: A tuple[peak, left, right] containing:
            peak: index where peak of data is
            left: index where the left bin of the signal is
            right: index where the right bin of the signal is

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
    breakpoint()

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

    
    seconds = np.arange(0, 100)

    for ii, fname in tqdm(enumerate(file_list[:5])):
        print(fname.stem)
        ADMXfile     = h5py.File(fname,'r')
        train_loader, maximum = read_loader(ADMXfile)
        sig_size = ADMXfile['injected'].attrs['sig_size']

        noise    = []
        denoised = []
        injected = []
        snri = []

        for i, batch in enumerate(train_loader):
            inputarr, targetarr = (batch[:batchsize], batch[batchsize:])

            input_seq  = torch.from_numpy(inputarr)
            input_seq  = input_seq.float().to(DEVICE)

            output_seq = model(input_seq).detach().cpu().numpy()
         
            denoised.append(output_seq.flatten())
            injected.append(targetarr.flatten())
            noise.append(inputarr.flatten())
            
        # 
        _, interval = get_snr(np.exp(np.mean(np.array(injected[:]), axis=0)) - 1)
        
        for i in seconds:
            # Scaling back to regular values and taking mean from index 0 to i + 1 for a total of 100 seconds
            noise_psd = np.exp(np.mean(np.array(noise[:i + 1]), axis=0)) - 1
            denoised_psd = np.exp(np.mean(np.array(denoised[:i + 1]), axis=0)) - 1

            noise_snr    = get_snr(noise_psd, interval)
            denoised_snr = get_snr(denoised_psd, interval)

            snri.append(denoised_snr[0] / noise_snr[0])
       
        plt.plot(seconds, snri, label=f'signal size = {sig_size}')
        plt.xlabel('Time (s)')
        plt.ylabel('SNRI')
        plt.ylim((0, 35))
        plt.legend(loc='upper right')
        plt.savefig(f'SNRI(int_time)_plot.pdf', dpi=300)

if __name__ == "__main__":
	main()