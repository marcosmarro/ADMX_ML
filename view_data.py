import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pyfftw
import matplotlib as mpl
from tqdm import tqdm
# mpl.rcParams['agg.path.chunksize'] = 1000

directory = Path('Training')
files = list(directory.glob('*.h5'))
# breakpoint()
file = "d:\Run1D_admx/7305182.egg"

# for file in files:
#     f = h5py.File(file)
#     try:
#         print(f['injected'].attrs['frequency'])
#     except KeyError:
#         print(f['injected'].attrs['frequencies'])
    

# breakpoint()
for i in range(40):
    with h5py.File(files[i], "r") as f:
        # print(files[0])
        # input = f['streams/stream0/acquisitions/0'][:].reshape(-1)
        # injected = f['injected'][:]
        input = f['input'][:]
        # denoised = f['denoised'][:][:int(2e5)]
        # print('amp:', f['injected'].attrs['amplitude'])
        # print('freq', f['injected'].attrs['frequency'])
        # print(f['injected'].attrs['log_sig_size'])
        # print(f['injected'].attrs['axion_energy_ueV'])
        # breakpoint()

    chunk = int(1e3)
    fft = pyfftw.interfaces.scipy_fft.rfft(input).real[1:]
    binned_fft = np.empty(int(len(fft) / chunk))
    for j in range(0, len(binned_fft)):
        binned_fft[j] = np.average(fft[chunk*j : chunk*j+chunk])
    plt.figure()
    plt.plot(binned_fft)
    plt.yscale('log')
    plt.savefig(f'plots/{i}.pdf')
    plt.close()
    # breakpoint()

# plt.plot(real_data)
# plt.plot(time, denoised, linewidth=0.5, c='g', label='denoised', alpha=0.5)
# plt.plot(injected, linewidth=0.5, c='r', label='injected', alpha=0.5)
plt.plot(input, linewidth=0.5, c='b', label='input', alpha=0.5)


plt.xlabel("Time")
plt.ylabel("Digitized Value")
# plt.ylim((124, 130))
# plt.legend()
plt.title("Time Series")
plt.tight_layout()
plt.show()