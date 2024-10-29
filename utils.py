import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
from tqdm import trange


# Generate noiseless s(t) data via eigenexpansion
def generate_samples(E, psi0, dt=1, nb=100):
    S = np.zeros(nb, dtype=np.complex128)
    for j in trange(nb):
        S[j] = np.sum(np.abs(psi0)**2 * np.exp(-1j * E * j * dt))
    return S


# Generate noiseless s(t) or any derivatives using functional form
def generate_samples_der(E, psi0, dt=1, nb=100, n=1):
    S = np.zeros(nb, dtype=np.complex128)
    for j in trange(nb): 
        S[j] = np.sum(np.abs(psi0)**2 * (-1j * E)**n * np.exp(-1j * E * j * dt))
    return S

# Cast the range of the spectrum
def lam2lamt(lam, lammin, lammax):
    lamt = np.pi / 2 * (lam - (lammin + lammax) / 2) / (lammax - lammin)
    return lamt

# Cast back the range of the spectrum
def lamt2lam(lamt, lammin, lammax):
    lam = lamt * 2 / np.pi * (lammax - lammin) + (lammin + lammax) / 2
    return lam

# Generate the reference state
def generate_phi(overlap, N):
    phi = np.zeros((N,1))
    phi[0] = np.sqrt(overlap)
    phi[1:] = np.sqrt((1 - phi[0]**2) / (N - 1))
    return phi

