import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
from tqdm import trange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.io import savemat

np.random.seed(999)
Tmaxs = np.array([1000])
shots = [50, 100, 500, 1000]
num_trajs = 100
for Tmax in Tmaxs:
    for n_shots in shots:
        overlap = 0.2  # 0: HF
        dt = 1

        def generate_samples(E, psi0, dt=1, nb=100):
            S = np.zeros(nb, dtype=np.complex128)
            for j in trange(nb):
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

        #data = scipy.io.loadmat('../../data/heh+_sto-3g_hamiltonian.mat')
        data = scipy.io.loadmat('../../data/Cr2_4000.mat')
        E = data['E']
        Et = lam2lamt(E, E[0], E[-1])

       
        phi = generate_phi(overlap, len(Et))
        print('generated phi')
        dataS = generate_samples(Et, phi, dt, Tmax)
        
        noisy_shots_dataS = []
        for i in range(num_trajs):
            noisydataS = np.zeros(dataS.shape)
            for k in range(dataS.shape[0]):
                mu = dataS[k]
                p = (1 + mu) / 2
                sample = np.random.binomial(n_shots, p.real)
                shifted_sample = 2 * sample - n_shots
                muapprox = (1 / n_shots) * shifted_sample
                noisydataS[k] = muapprox
            noisy_shots_dataS.append(noisydataS)
        
        #avg_noisy_shots_dataS = np.mean(np.array(noisy_shots_dataS), axis=0)
        noisy_shots_dataS = np.array(noisy_shots_dataS)
        t = np.arange(0, Tmax, 1)

     
        X = t[:, np.newaxis]
        y_real = noisy_shots_dataS.real
        kernel = C(1.0, (1e-6, 1e1)) * RBF(1.0, (1e-4, 1e1))
        gp_real = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10)
        gp_real.fit(X, y_real)
        mean_prediction, _ = gp_real.predict(X, return_std=True)

        # Save the denoised signal
        savemat('../../matlab/denoised_data/denoised_dataS_Cr2_GP_Tmax=' + str(Tmax) + '_overlap=' + str(overlap) + 'Nshots=' + str(n_shots) + '_real.mat', {'denoised_dataS': mean_prediction})
