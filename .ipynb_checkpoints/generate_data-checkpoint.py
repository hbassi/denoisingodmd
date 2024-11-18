import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
import argparse
import utils as ut

def main(molecule, noise, Tmax, overlap, dt, num_trajs):
    np.random.seed(999)
    
    # Load the data
    if molecule == 'Cr2':
        data = scipy.io.loadmat(f'./data/{molecule}_4000.mat')
    elif molecule == 'LiH':
        data = scipy.io.loadmat(f'./data/{molecule}_2989.mat')
    elif molecule == 'H6':
        data = scipy.io.loadmat(f'./data/{molecule}4000.mat')
    psiHF = data['psiHF']
    E = data['E']
    Et = ut.lam2lamt(E, E[0], E[-1])
    
    # Generate the reference state
    phi = ut.generate_phi(overlap, len(Et))
    print('generated phi')
    dataS = ut.generate_samples_der(Et, phi, dt, Tmax, n=0)

    # Save the noiseless data
    noiseless_filename = f'noiseless_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}.npy'
    with open(noiseless_filename, 'wb') as f:
        np.save(f, dataS)

    # Generate the noisy data
    tdataS = []
    for i in range(num_trajs):
        ndataS = (dataS + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax)).real
        clipped = np.clip(ndataS, -1, 1)
        # We know the value at time t = 0 is 1
        clipped[0] = 1.0  
        tdataS.append(clipped)
    if num_trajs == 1:
        tdataS = tdataS[0].T
    # Save the noisy time series
    noisy_filename = f'noisy_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}.npy'
    with open(noisy_filename, 'wb') as f:
        np.save(f, tdataS)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process parameters for data generation.")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level.")
    parser.add_argument("--Tmax", type=int, default=1000, help="Final time.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap with the GS.")
    parser.add_argument("--dt", type=int, default=1, help="Time step.")
    parser.add_argument("--num_trajs", type=int, default=1, help="Number of noisy trajectories.")
    
    args = parser.parse_args()
    main(args.molecule, args.noise, args.Tmax, args.overlap, args.dt, args.num_trajs)
