import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
import argparse
import utils as ut
from tqdm import trange

def main(molecule, noise, Tmax, overlap, dt, num_trajs, numpad):
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
    Et = ut.lam2lamt(E, E[0] + 0.2, E[-1] - 0.2)
    # E_center = E - (E.min() + E.max())/2
    # Et = E_center / np.abs(E_center).max()
    #Et = E / 170
    # Generate the reference state
    phi = ut.generate_phi(overlap, len(Et))
    print('generated phi')
    
    # Save absolute errors of estimated GSE and ground truth GSE
    errors = []
    dataS = ut.generate_samples_der(Et, phi, dt, Tmax, n=0)

    # Generate the noisy data
    ndataS = (dataS + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax))
    clipped = np.clip(ndataS, -1, 1)
    # We know the value at time t = 0 is 1
    clipped[0] = 1.0  

    
    # Generate individual time series up until Tmax
    for i in trange(15, Tmax):
    
        
        # Use FFT with zero-padding to estimate GSE
        gse_estimate = ut.specest(clipped[:i].real, numpad)
        gt_gse = Et[0]
        # Compute the absolute error 
        abs_err = np.abs(np.abs(gse_estimate) - np.abs(gt_gse.item()))
        print('GSE Estimate: ', gse_estimate)
        print('GT GSE: ' , gt_gse.item())
        print('Absolute error: ', abs_err)
        print('================================================')
        
        errors.append(abs_err)
    
    # Plot error results
    with open('./figures/FFT_cr2_gse_errors_numpad='+str(numpad)+'_noise='+str(noise)+'_multiple_pad_ff_left_right_absolute_errors_real.npy', 'wb') as f:
        np.save(f, errors)
    plt.semilogy(np.arange(15, Tmax), errors, '*')
    plt.plot([15, Tmax], [1e-3, 1e-3], 'k:')
    plt.xlabel('# observables')
    plt.ylabel('absolute error')
    plt.ylim([1e-7, 1e-1])
    plt.title('Numpad = '+str(numpad) + ', noise = ' + str(noise))
    plt.savefig('./figures/FFT_cr2_gse_errors_numpad='+str(numpad)+'_noise='+str(noise)+'_multiple_pad_ff_left_right_real.png')
        
    

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process parameters for data generation.")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level.")
    parser.add_argument("--Tmax", type=int, default=1000, help="Final time.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap with the GS.")
    parser.add_argument("--dt", type=int, default=1, help="Time step.")
    parser.add_argument("--num_trajs", type=int, default=1, help="Number of noisy trajectories.")
    parser.add_argument("--numpad", type=int, default=1, help="Length of zero-padding to include in FFT")
    
    args = parser.parse_args()
    main(args.molecule, args.noise, args.Tmax, args.overlap, args.dt, args.num_trajs, args.numpad)
