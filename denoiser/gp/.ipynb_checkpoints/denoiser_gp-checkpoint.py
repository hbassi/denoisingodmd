import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
from tqdm import trange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Hyperparameter, Kernel
from scipy.io import savemat
from tqdm import trange
from scipy.spatial.distance import cdist

np.random.seed(999)
Tmaxs = np.arange(20, 1000, 10)
noises = [0.1]

for k in trange(len(Tmaxs)):
    for j in range(len(noises)):
        overlap = 0.2  
        dt = 1
        noise = noises[j]
        Tmax = Tmaxs[k]
        def generate_samples(E, psi0, dt=1, nb=100):
            S = np.zeros(nb, dtype=np.complex128)
            for j in range(nb):
                S[j] = np.sum(np.abs(psi0)**2 * np.exp(-1j * E * j * dt))
            return S

        def lam2lamt(lam, lammin, lammax):
            lamt = np.pi / 2 * (lam - (lammin + lammax) / 2) / (lammax - lammin)
            return lamt

        def lamt2lam(lamt, lammin, lammax):
            lam = lamt * 2 / np.pi * (lammax - lammin) + (lammin + lammax) / 2
            return lam

        def generate_phi(overlap, N):
            phi = np.zeros((N, 1))
            phi[0] = np.sqrt(overlap)
            phi[1:] = np.sqrt((1 - phi[0]**2) / (N - 1))
            return phi

        data_cr2 = scipy.io.loadmat('../../data/Cr2_4000.mat')
        psiHF = data_cr2['psiHF']
        E = data_cr2['E']
        Et = lam2lamt(E, E[0], E[-1])

        if overlap == 0:
            dataS = generate_samples(Et, psiHF, dt, Tmax)
            phi = psiHF
        else:
            phi = generate_phi(overlap, len(Et))
            dataS = generate_samples(Et, phi, dt, Tmax)

        # tdataS = []
        # num_trajs = 100
        # for i in range(num_trajs):
        #     ndataS = (dataS + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax))
        #     tdataS.append(ndataS)
        # noisydataS = np.array(tdataS)
        tdataS = dataS + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax)
        tdataS_real = np.clip(tdataS.real, -1, 1)

        t = np.arange(0, Tmax, 1)
       
        X = t[:, np.newaxis]
        y_real = tdataS_real
        #kernel = C(1.0, (1e-6, 1e1)) * RBF(1.0, (1e-4, 1e1))
        kernel = C(1.0, (1e-6, 1e1)) * RBF(1.0, (1e-4, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-4, 1e2))+ WhiteKernel(noise_level=noise, noise_level_bounds=(noise/100, 10*noise))
        #kernel = C(1.0, (1e-6, 1e1))  * ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-4, 1e2)) + WhiteKernel(noise_level=noise, noise_level_bounds=(noise/100, 10*noise))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, alpha=1e-5)
        gp.fit(X, y_real)
        mean_prediction = gp.predict(X, return_std=False)
    
        mean_prediction[0] = 1

    
        savemat('../../matlab/denoised_data/denoised_dataS_Cr2_GP_Tmax=' + str(Tmax) + '_overlap=' + str(overlap) + 'noise=' + str(noise) + '_real_multi_traj.mat', {'denoised_dataS': mean_prediction})
