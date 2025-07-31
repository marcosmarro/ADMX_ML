import h5py
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from injected import make_injected
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Digitizes the data and injects a classical random field")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Data directory where data files are stored')
parser.add_argument('--type', '-t', type=str, default='Training', help='Type of file to inject')
args = parser.parse_args()


def digitize_data(data):
    "Takes in an array and digitizes it to 256 values"

    normalized = (data + 4) / (8)
    digitized = np.clip(normalized, 0, 1)
    digitized = np.round(normalized * 2047)
    
    return digitized


def main():
    '''
    Runs the script in the following order:
        1. Reads the pure noise .egg files 
        2. Generates an injected signal
        3. Digitizes both the injected signal and noise+injected
        3. Writes a .h5 file with two channels:
            'input': the input signal for training (noise + injected)
            'injected': the injected/target signal for training 
    '''

    directory = Path('D:/Run1D_admx/corrected_data/Training') # Training/Science

    files = sorted(directory.glob('*.h5'))

    np.random.seed(42)
    size = 10 ** np.random.uniform(25.5, 26, size=len(files))
    # e_a0_ueV = np.round(np.random.uniform(4, 6, size=len(files)), decimals=3)
    # f_detune = np.round(np.random.triangular(-9, 0, 9, size=len(files)), decimals=3) * 1e4
    amplitude = np.round(np.random.uniform(0.15, 0.35, size=len(files)), decimals=3)
    amplitude_2 = np.round(np.random.uniform(0.15, 0.35, size=len(files)), decimals=3)
    frequency_1 = np.round(np.random.uniform(9.5e4, 1.05e5, size=len(files)), decimals=3) 
    frequency_2 = np.round(np.random.uniform(1.7e5, 1.8e5, size=len(files)), decimals=3)
    frequency = np.concatenate((frequency_1[:15], frequency_2[:15]))
    # 100_000 - 174_950
    breakpoint()
    for i, filename in tqdm(enumerate(files[:40])):
        raw_noise = h5py.File(filename, "r")['Data'][:].real
        raw_noise = np.append(raw_noise, [0, 0])

        if i < 30:
            raw_injected = make_injected(len(raw_noise), frequency[i], amplitude[i])
        else:
            raw_injected = make_injected(len(raw_noise), frequency_1[i], amplitude[i]) + make_injected(len(raw_noise), frequency_2[i], amplitude_2[i])
        # raw_injected = np.append(raw_injected, [0, 0])

        input_data = digitize_data(raw_noise + raw_injected)
        injected_data = digitize_data(raw_injected)

        # import pyfftw
        # injected_fft = pyfftw.interfaces.scipy_fft.rfft(injected_data).real[1:] ** 2
        # noise_fft = pyfftw.interfaces.scipy_fft.rfft(input_data).real[1:] ** 2
        # plt.plot(injected_fft, alpha=0.5, label='injected')
        # plt.plot(noise_fft, alpha=0.5, label='noise')
        # # plt.plot(input_data)
        # plt.legend()
        # plt.yscale('log')
        # plt.show()
        # breakpoint()
       
        f = h5py.File(f'Training/{filename.stem}_training.h5', 'w') # change to training/science for different needs
        f.create_dataset('input', data=input_data)
        f.create_dataset('injected', data=injected_data)
        # f['injected'].attrs['log_sig_size'] = f'{size[i]:.3e}'
        # f['injected'].attrs['axion_energy_ueV'] = e_a0_ueV[i]
        if i < 30:
            f['injected'].attrs['frequency'] = frequency[i]
            f['injected'].attrs['amplitude'] = amplitude[i]
        else:
            f['injected'].attrs['frequencies'] = [frequency_1[i], frequency_2[i]]
            f['injected'].attrs['amplitudes'] = [amplitude[i], amplitude_2[i]]


        print(f'{filename.name} injected')
        

if __name__ == '__main__':
    main()