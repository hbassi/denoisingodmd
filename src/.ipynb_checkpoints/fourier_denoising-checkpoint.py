import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils as ut
import math
x = np.load('./data/noisy_dataS_Cr2_noise=0.8_Tmax=1500_overlap=0.2_dt=1.0_ff=0.2_left_right.npy')
x_less = np.load('./data/noisy_dataS_Cr2_noise=0.1_Tmax=1500_overlap=0.2_dt=1.0_ff=0.2_left_right.npy')
x_true = np.load('./data/noiseless_dataS_Cr2_noise=0.01_Tmax=1500_overlap=0.2_dt=1.0_ff=0.2_left_right.npy')

# FFT
X_fft = scipy.fft.fft(x[:].real)
X_fft_shifted = scipy.fft.fftshift(X_fft)
frequencies = scipy.fft.fftshift(scipy.fft.fftfreq(len(x), d=1))[:]


data = loadmat('./data/Cr2_4000.mat')
E = data['E']
lamt = ut.lam2lamt(E, E[0] - 0.2, E[-1] + 0.2)
# Thresholding
gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
for gamma in gammas:
    threshold_factor = gamma
    threshold = threshold_factor * np.median(np.abs(X_fft_shifted))
    # Zero out frequencies below the threshold
    X_fft_filtered = X_fft_shifted * (np.abs(X_fft_shifted) > threshold)
    # IFFT to obtain the denoised signal
    x_denoised = scipy.fft.ifft(scipy.fft.ifftshift(X_fft_filtered))
    # We know s(0) = 1 
    x_denoised[0] = 1
    # MAE
    print('MAE (GT): ', np.mean(np.abs(x_denoised.real - x_true.real)))
    print('MAE (N): ', np.mean(np.abs(x.real - x_denoised.real)))
    print('MAE (N-GT): ', np.mean(np.abs(x.real - x_true.real)))
    denoised_filename = f'./data/denoised_dataS_Cr2_noise=0.8_Tmax=1500_overlap=0.2_dt=1.0_ff=0.2_left_right_threshold={threshold_factor}.npy'
    with open(denoised_filename, 'wb') as f:
        np.save(f, x_denoised)
