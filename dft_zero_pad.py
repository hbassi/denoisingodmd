import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
import argparse
import utils as ut
from tqdm import trange

def main(molecule, noise, Tmax, overlap, dt, numpad, option='', denoised=False):
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
    Et = ut.lam2lamt(E, E[0] - 0.2, E[-1] + 0.2)

    
    if not denoised:
        print('Using noisy data')
        # Generate the reference state
        phi = ut.generate_phi(overlap, len(Et))
        print('generated phi')

       
        dataS = ut.generate_samples_der(Et, phi, dt, Tmax, n=0)

        # Generate the noisy data
        ndataS = (dataS + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax))
        clipped = np.clip(ndataS, -1, 1)
        # We know the value at time t = 0 is 1
        clipped[0] = 1.0
        savename = f'FFT_zero_padding_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}'
    
    else:
        denoised_filename = f'denoised_dataS_Cr2_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
        print('Using denoised data')
        clipped = np.load('./data/'+denoised_filename)
        savename = f'fourier_denoised_zero_padding_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'
        
    # Save absolute errors of estimated GSE and ground truth GSE
    errors = []
    gt_gse = E[0]
     # Generate individual time series up until Tmax
    for i in trange(20, Tmax + 1):
    
        
        # Use FFT with zero-padding to estimate GSE
        gse_estimate = ut.specest(clipped[:i].real, numpad)
        gse_estimate_cast = ut.lamt2lam(gse_estimate, E[0] - 0.2, E[-1] + 0.2)
        # Compute the absolute error 
        #relative_error = np.abs(np.abs(gse_estimate) - np.abs(gt_gse.item())) / np.abs(gt_gse.item())
        abs_error = np.abs(np.abs(gse_estimate_cast) - np.abs(gt_gse.item()))
        print('GSE Estimate: ', gse_estimate_cast)
        print('GT GSE: ' , gt_gse.item())
        #print('Relative error: ', relative_error)
        print('Absolute error: ', abs_error)
        print('================================================')
        
        #errors.append(relative_error)
        errors.append(abs_error)
    
    # Plot error results
    with open('./figures/'+savename+'_padding='+str(numpad)+'_ff_left_right_absolute_errors_real_test1.npy', 'wb') as f:
        np.save(f, errors)
    plt.semilogy(np.arange(20, Tmax + 1), errors, '*')
    plt.plot([20, Tmax], [1e-3, 1e-3], 'k:')
    plt.xlabel('# observables')
    plt.ylabel('absolute error')
    plt.ylim([1e-7, 1e-1])
    plt.title('Numpad = '+str(numpad) + ', noise = ' + str(noise))
    plt.savefig('./figures/'+savename+'_gse_errors_numpad='+str(numpad)+'_padding_ff_left_right_real_test1.png')
        
    

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process parameters for data generation.")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level.")
    parser.add_argument("--Tmax", type=int, default=1000, help="Final time.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap with the GS.")
    parser.add_argument("--dt", type=int, default=1, help="Time step.")
    parser.add_argument("--numpad", type=int, default=1, help="Length of zero-padding to include in FFT")
    parser.add_argument("--option", type=str, default='', help="Which kind of energy casting.")
    parser.add_argument("--denoised", type=lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"], default=True, help="Denoised data from FD or not")
    
    args = parser.parse_args()
    main(args.molecule, args.noise, args.Tmax, args.overlap, args.dt, args.numpad, args.option, args.denoised)
