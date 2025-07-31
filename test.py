import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse
import lwpt
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory
parser.add_argument('--denoising_model', '-m', type=str, default='fcnet', help='Denoising model we would like to train [fcnet/punet/transformer/lwpt] (Default: punet).')

args = parser.parse_args()

batchsize = 1
input_size = 40000

def digitize_data(data):
    "Takes in an array and digitizes it to 256 values"

    normalized = (data + 4) / (8)
    digitized = np.clip(normalized, 0, 1)
    digitized = np.round(normalized * 2047)
    
    return digitized

def read_loader(ADMXfile):
    train = digitize_data(ADMXfile['streams/stream0/acquisitions/0'][:].reshape(-1)) - 1023
    max_index = int(4e7)
    train = train[:max_index].reshape(-1, batchsize, input_size)

    return train

directory = Path('D:\Run1D_admx')
file_list = list(directory.glob('*.egg'))

for fname in file_list:

    if args.denoising_model == 'fcnet':
        model = torch.load(f'FCNet.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == 'punet':
        model = torch.load(f'PUNet.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == 'transformer':
        model = torch.load(f'Transformer.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    elif args.denoising_model == 'lwpt':
        model = torch.load(f'LWPT.pth', map_location=DEVICE, weights_only=False)
        model.eval()

    ADMXfile = h5py.File(fname,'r')
    noise = []
    denoised = []
    train_loader = read_loader(ADMXfile)
    breakpoint()
    for i, batch in tqdm(enumerate(train_loader)):
        inputarr=  batch[:batchsize]
    
        input_seq = torch.from_numpy(inputarr)
        # Forward pass

        if args.denoising_model == 'fcnet':
            input_seq = input_seq.float().to(DEVICE)
            output_seq = model(input_seq).detach().cpu().numpy()
        elif args.denoising_model == 'lwpt':
            input_seq = input_seq.unsqueeze(0).double().to(DEVICE)
            output_seq = model(input_seq).detach().cpu().numpy()
        else:
            input_seq = input_seq.long().to(DEVICE)
            output_seq = model(input_seq).argmax(dim=1).detach().cpu().numpy()

        denoised.append(np.round(output_seq.flatten()))
        noise.append(inputarr.flatten())
        # breakpoint()

    denoised = np.concatenate(denoised, axis=0)
    noise = np.concatenate(noise, axis=0)
    print(denoised)
    # breakpoint()
    with h5py.File(f'{fname.stem}_denoised_{args.denoising_model}.h5', 'w') as f:
        f.create_dataset('denoised', data=denoised)
        f.create_dataset('input', data=noise)


