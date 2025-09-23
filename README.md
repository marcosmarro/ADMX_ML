# ADMX: PSD Denoising for Signal Detection

A Python machine learning pipeline for denoising **power spectral density (PSD)** data derived from time-series measurements. This project is designed to recover weak injected signals (such as simulated axion signals) that are buried under noise.

The repo covers signal simulation, preprocessing, training, inference, and evaluation, providing a full pipeline for PSD-based signal recovery.

## Features

- **Signal Simulation**: Generate synthetic axion-like signals in noisy time-series data.
- **Injection & Digitization**: Combine noise with simulated signals and digitize the data for training.
- **Neural Networks**: Train deep learning architectures (defined in `network.py`) for PSD denoising.
- **SNR Evaluation**: Quantitatively assess denoising performance with signal-to-noise ratio (SNR) metrics.

---

## Repository Structure

```
ADMX_ML/
├── train.py                 # Model training script
├── inference.py             # Apply trained model to denoise PSDs
├── frequency_simulation.py  # Simulate axion-like time-series signal
├── inject_and_digitize.py   # Inject simulated signal + noise, digitize time-series
├── snr_evaluation.py        # Compute SNR metrics and analysis
├── network.py               # PyTorch neural network architectures
├── requirements.txt         # Python dependencies
├── Training/                # Training FITS files (input data)
├── Science/                 # Raw science time to be denoised
├── DenoisedScience/         # Output denoised PSDs
└── README.md
```

---

## Installation & Setup

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/marcosmarro/ADMX_ML.git
   cd ADMX_ML
   ```

2. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv venv          # Mac/Linux
   source venv/bin/activate
   ```
   ```bash
   python -m venv venv          # Windows
   venv\Scripts\activate       
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

**Note:** Requirements.txt doesn't include PyTorch + CUDA. For CUDA support in Windows/Linux, visit https://pytorch.org/get-started/locally/ to set up.
---

## Usage

**Repo doesn't include Training/Science files, only includes trained models. Skip to step 4 for evaluation.**

### 1. Train the Model

Train a network on the prepared training set:

```bash
python train.py -d [directory] -m [model]
```

- [directory] is where all the training files live. Defaults to Training/
- [model] is the model wished to be trained. Must choose from `[unet/dae]`

### 2. Denoise Science Data

Denoise science PSDs with trained model:

```bash
python inference.py -d [directory] -m [model]
```

- [directory] is where all the science files live. Defaults to Science/
- [model] is the denoising model wished to be used. Must choose from `[unet/dae]`

### 3. Evaluate SNR

Run SNR analysis on denoised outputs:

```bash
python snr_evaluation.py -d [directory] -m [model]
```

- [directory] is where all the denoised science files live. Defaults to Denoised_science/
- [model] is the model wished to be evaluated. Must choose from `[unet/dae]`

### 4. Check Data and Noise Metrics

Open the Jupyter Notebooks, data_metrics.ipynb & noise_metrics.ipynb, to plot different results for injected and uninjected signals, respectively.
---

## Neural Network Architectures

All model definitions are in `network.py`. Architectures explored include:

* UNet: arXiv:1505.04597
* Denoising Autoencoder (DAE): arXiv:2205.13513

---

## Contact

**Author**: Marcos Marroquin  
**Institution**: University of Washington  
**Email**: marcosvmarro@gmail.com  