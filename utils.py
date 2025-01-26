import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
from tqdm import trange


# Generate noiseless s(t) data via eigenexpansion
def generate_samples(E, psi0, dt=1, nb=100):
    S = np.zeros(nb, dtype=np.complex128)
    for j in range(nb):
        S[j] = np.sum(np.abs(psi0)**2 * np.exp(-1j * E * j * dt))
    return S


# Generate noiseless s(t) or any derivatives using functional form
def generate_samples_der(E, psi0, dt=1, nb=100, n=0):
    S = np.zeros(nb, dtype=np.complex128)
    for j in range(nb): 
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

# Use zero padding and FFT to estimate dominant frequency
def specest(data, numpad, delta_t):
    n = len(data)
    # Zeropad the sequence
    x = np.concatenate([data, np.zeros(numpad)])
    N = len(x)
    # FFT
    xhat = scipy.fft.fftshift(scipy.fft.fft(x))
    freq = np.linspace(-np.pi, np.pi, N)
    # Retrieve most dominant frequency
    ind = np.argsort(np.abs(xhat))
    is_symmetric = np.allclose(np.abs(xhat), np.abs(xhat)[::-1], atol=1e-4)
    if is_symmetric:
        return freq[min(ind[-1], ind[-2])] / delta_t
    else:
        return freq[ind[-1]] / delta_t

#Helper function for data Hankelization
def make_hankel(data, m, n):
    return hankel(data[:m], data[m-1:m+n-1])
    
# DMD
def dmd(data, dt, tol=1e-6):
    k = len(data)
    X = make_hankel(data, int(np.floor(k / 2)), int(np.ceil(1 / 2 * k)))

    X1 = X[:, :-1]
    X2 = X[:, 1:]

    U, S, Vh = svd(X1, full_matrices=False)
    V = Vh.conj().T
   
    r = np.sum(S > tol * S[0])
    U = U[:, :r]
    S = S[:r]
    V = V[:, :r]

    Atilde = U.conj().T @ X2 @ V / S
    mu = eig(Atilde)[0]
    omega = np.log(mu) / dt

    return omega

#  #omega = dmd(dataS[:t[i]], dt, tol[j])
#Helper function to run DMD for various time
def run_compare(dataS, dt, tol=[1e-1, 1e-2, 1e-3], Tmax=500, step=10):
    t = np.arange(20, Tmax + 1, step)
    lam = np.inf * np.ones((len(t), len(tol), 2))  # Store two largest values
    cond_nums = np.inf * np.ones((len(t), len(tol)))
    for i in trange(len(t)):
        for j in range(len(tol)):
            omega = MODMD(dataS[:, :t[i]], dt, tol[j])
            imag_parts = np.imag(omega)  # Extract imaginary parts

            # Sort in descending order and take the two largest values
            sorted_imag = np.sort(imag_parts)[::-1]
            top_two = sorted_imag[:2] if len(sorted_imag) >= 2 else [sorted_imag[0], np.nan]

            lam[i, j, 0] = -top_two[0]  # First largest
            lam[i, j, 1] = -top_two[1]  # Second largest

    return lam, t

# #Helper function to run DMD for various time
# def run_compare(dataS, dt, tol=[1e-1, 1e-2, 1e-3], Tmax=500, step=10):
#     t = np.arange(20, Tmax + 1, step)
#     lam = np.inf * np.ones((len(t), len(tol)))
    
#     for i in trange(len(t)):
#         for j in range(len(tol)):
#             omega = dmd(dataS[:t[i]], dt, tol[j])
#             lam[i, j] = -np.max(np.imag(omega))
            
#     return lam, t


def plot_compare(t, lam, tol, E, mytitle='', xlimits=None, ylimits=None, savename=''):
    plt.figure()
    marker = '*ods'
    lgnd = []
    abs_errs = []
    for i in range(len(tol)):
        mark = marker[i % len(marker)]
        err = np.abs(lam[:, i, 0] - E[0])
        abs_errs.append(err)
        plt.semilogy(t, err, mark, label=f'tol = {tol[i]}')

    print('Errors: ', abs_errs)
    plt.plot([0, t[-1]], [1e-3, 1e-3], ':k')
    plt.legend()
    plt.xlabel('# observables')
    plt.ylabel('absolute error')
    if mytitle:
        plt.title(mytitle)
    if xlimits:
        plt.xlim(xlimits)
    if ylimits:
        plt.ylim(ylimits)

    # Plot error results
    
    plt.savefig('./figures/'+savename+'_test_E0.png')
    with open('./figures/'+savename+'_ODMD.npy', 'wb') as f:
        np.save(f, abs_errs)
    plt.close()
    
#     plt.figure()
#     marker = '*ods'
#     lgnd = []
#     abs_errs = []
#     for i in range(len(tol)):
#         mark = marker[i % len(marker)]
#         err = np.abs(lam[:, i, 1] - E[1])
#         abs_errs.append(err)
#         plt.semilogy(t, err, mark, label=f'tol = {tol[i]}')

#     print('Errors: ', abs_errs)
#     plt.plot([0, t[-1]], [1e-3, 1e-3], ':k')
#     plt.legend()
#     plt.xlabel('# observables')
#     plt.ylabel('absolute error')
#     if mytitle:
#         plt.title(mytitle)
#     if xlimits:
#         plt.xlim(xlimits)
#     if ylimits:
#         plt.ylim(ylimits)
   
    
#     plt.plot([0, t[-1]], [1e-3, 1e-3], ':k')
#     plt.legend()
#     plt.xlabel('# observables')
#     plt.ylabel('absolute error')
#     if mytitle:
#         plt.title(mytitle)
#     if xlimits:
#         plt.xlim(xlimits)
#     if ylimits:
#         plt.ylim(ylimits)
#     plt.savefig('./figures/'+savename+'_test_E1.png')
    
def BHankel(data):
    """
    Construct a block Hankel matrix from input data.
    Parameters:
    - data: Input data matrix (rows are variables, columns are time snapshots)
    - m: Block size for the Hankel matrix
    Returns:
    - H: Block Hankel matrix
    """
    rows, cols = data.shape
    m = cols//2
    # Initialize the Hankel matrix
    X = np.zeros((rows * m, cols - m), dtype=complex)
    # Loop through each column and fill the matrix
    for i in range(cols - m):
        temp = data[:, i:i + m].T
        X[:, i] = temp.ravel()  # Flatten temp and store in X
    return X


def MODMD(data, dt, tol=1e-8, eigid=0):
    """
    Dynamic Mode Decomposition (DMD) with SVD truncation
    Parameters:
    - data: Input data matrix (each column is a snapshot in time)
    - dt: Time between data snapshots
    - tol: Tolerance for rank truncation
    - eigid: Number of eigenvalues to return
    Returns:
    - E_approx: Approximate sorted imaginary part of eigenvalues
    """
    # Shifted data matrices (block Hankel)
    
    X = BHankel(data)
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    # SVD of X1
    U, S, Vh = svd(X1, full_matrices=False)
    # Rank truncation
    r = np.sum(S > tol * S[0])
    U = U[:, :r]
    S = S[:r]
    V = Vh[:r, :].conj().T
    S_inv = np.diag(1/S)
    # DMD computation
    Atilde = np.dot(np.dot(U.T, X2), np.dot(V, S_inv))
   
    # Eigenvalue computation
    mu = eig(Atilde)[0]
    omega = np.log(mu) / dt
    #Etilde = np.sort(-np.imag(omega))
    # Return the approximate eigenvalues
    return omega #Etilde[eigid]