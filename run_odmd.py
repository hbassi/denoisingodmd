import numpy as np
import utils as ut
import argparse
import scipy

seed = 999
np.random.seed(seed)

def main(molecule, noise, Tmax, overlap, dt, tol, step, fudge_factor=0, option='', baseline=True):
    # File names for the loading of corresponding data from the generation script or denoising notebook
    noiseless_filename = f'noiseless_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    noisy_filename = f'noisy_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    denoised_filename = f'denoised_dataS_Cr2_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    
    # Depending on baseline ODMD or denoising ODMD load in the data
    if baseline and noise != 0:
        print('Using noisy data')
        dataS = np.load('./data/'+noisy_filename)
        savename = f'baseline_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}'
    elif baseline and noise == 0:
        print('Using noiseless data')
        dataS = np.load('./data/'+noiseless_filename)
        savename = f'baseline_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}'
    else:
        print('Using denoised data')
        dataS = np.load('./data/'+denoised_filename)
        savename = f'fourier_denoised_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}'
    
    # Run ODMD
    lamt,t = ut.run_compare(dataS,dt,tol,Tmax,step)
    
    # Load in the spectrum to recast the results for plotting
    if molecule == 'Cr2':
        data = scipy.io.loadmat(f'./data/{molecule}_4000.mat')
    elif molecule == 'LiH':
        data = scipy.io.loadmat(f'./data/{molecule}_2989.mat')
    elif molecule == 'H6':
        data = scipy.io.loadmat(f'./data/{molecule}4000.mat')
    E = data['E']
    
    # Recast the data for plotting
    lam = ut.lamt2lam(lamt, E[0] - fudge_factor, E[-1] + fudge_factor)
    
    # Plot absolute error using recasted spectrum and estimates
    ut.plot_compare(
        t=t,
        lam=lam,
        tol=tol,
        E=E,  
        mytitle=molecule + ' (noise = '+ str(noise) + ')',
        xlimits=(20, Tmax),
        ylimits=[1e-8,10],
        savename=savename
    )
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process parameters for data generation.")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level.")
    parser.add_argument("--Tmax", type=int, default=1000, help="Final time.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap with the GS.")
    parser.add_argument("--dt", type=int, default=1, help="Time step.")
    parser.add_argument("--tol", type=float, nargs='+', default=[0.1], help="SVD tolerances.")
    parser.add_argument("--step", type=int, default=1, help="Successive length.")
    parser.add_argument("--fudge_factor", type=float, default=1.0, help="Left and right endpoint fudge factor.")
    parser.add_argument("--option", type=str, default='', help="Which kind of energy casting.")
    parser.add_argument("--baseline", type=lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"], default=True, help="Baseline ODMD or not")

    
    args = parser.parse_args()

    main(args.molecule, args.noise, args.Tmax, args.overlap, args.dt, args.tol, args.step, args.fudge_factor, args.option, args.baseline)
