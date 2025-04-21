import numpy as np
import utils as ut
import argparse
import scipy

seed = 999
np.random.seed(seed)

def main(molecule, noise, Tmax, overlap, dt, tol, step, fudge_factor=0, option='', baseline=True, stacked=False, depolarized=0):
    # File names for the loading of corresponding data from the generation script or denoising notebook
    noiseless_filename = f'noiseless_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    noisy_filename = f'noisy_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    denoised_filename = f'denoised_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    stacked_filename = f'stacked_noisy_denoised_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'.npy'
    depolarized_filename = f'noisy_dataS_{molecule}_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_'+option+'_depolarized_gamma='+str(depolarized)+'.npy'
    # Depending on baseline ODMD or denoising ODMD load in the data
    if baseline and noise != 0 and depolarized == -1:
        
        print('Using noisy data')
        dataS = np.load('./data/'+noisy_filename)
        dataS = dataS.reshape((1, dataS.shape[0]))
        savename = f'{molecule}_baseline_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_real'
    elif baseline and depolarized != -1:
        print('Using depolarized noisy data')
        dataS= np.load('./data/'+depolarized_filename)
        dataS = dataS.reshape((1, dataS.shape[0]))
        savename = f'{molecule}_baseline_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_real)_depolarized_gamma={depolarized}'
    # elif baseline and noise == 0:
    #     print('Using noiseless data')
    #     dataS = np.load('./data/'+noiseless_filename)
    #     savename = f'{molecule}_baseline_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_real'
    
    else:
        if baseline and stacked and noise == 0.0:
            print('Using stacked baseline noiseless')
            dataS = np.load('./data/stacked_noiseless_dataS_Cr2_noise=0.001_Tmax=1500_overlap=0.2_dt=1.0_ff=0.2_left_right.npy')
            savename = f'{molecule}_baseline_no_stacked_odmd_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_tol={tol[0]}-{tol[-1]}_real'
        elif not stacked:
            print('Using denoised data only')
            dataS = np.load('./data/'+denoised_filename)
            savename = f'{molecule}_fourier_denoised_odmd_3mode_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_real'
        else:
            print('Using stacked denoised and noisy data')
            dataS = np.load('./data/'+stacked_filename)
            savename = f'{molecule}_fourier_denoised_odmd_6_stacked_denoised_noise={noise}_Tmax={Tmax}_overlap={overlap}_dt={dt}_tol={tol[0]}-{tol[-1]}_real'
    print(dataS.shape)
    dataS = dataS[:, :]#.reshape((1, dataS.shape[1]))
    noiseless_dataS = np.load('./data/stacked_noiseless_dataS_Cr2_noise=0.001_Tmax=1500_overlap=0.2_dt=1.0_ff=0.2_left_right.npy')
    noiseless_dataS = noiseless_dataS[:, :]
    print(dataS.shape)
    # Run ODMD
    #lamt,t, cond_nums = ut.run_compare(dataS.real,dt,tol,Tmax,step)
    lamt,t, retSs = ut.run_compare(dataS.real,dt,tol,Tmax,step,extra=None)
    #import pdb; pdb.set_trace()
    real_lamt = np.real(lamt)
    imag_lamt = np.imag(lamt)
    
    # Load in the spectrum to recast the results for plotting
    if molecule == 'Cr2':
        data = scipy.io.loadmat(f'./data/{molecule}_4000.mat')
    elif molecule == 'LiH':
        data = scipy.io.loadmat(f'./data/{molecule}_2989.mat')
    elif molecule == 'H6':
        data = scipy.io.loadmat(f'./data/{molecule}4000.mat')
    E = data['E']
    
    # Recast the data for plotting
    lam = ut.lamt2lam(imag_lamt, E[0] - fudge_factor, E[-1] + fudge_factor)
    with open('./figures/'+savename+'_eigenvalues.npy', 'wb') as f:
        np.save(f, real_lamt + 1j *lam)
    f.close()
    with open('./figures/svs/'+savename+'_cond_numA.npy', 'wb') as f:
        np.save(f, np.array(retSs))
    # with open('./figures/svs/'+savename+'_noiseless_norm.npy', 'wb') as f:
    #     np.save(f, norms)
    
    # Plot absolute error using recasted spectrum and estimates
    ut.plot_compare(
        t=t,
        lam=lam,
        tol=tol,
        E=E,  
        mytitle=molecule + ' (noise = '+ str(noise) + ')',
        xlimits=(20, Tmax),
        ylimits=[1e-7,1e-1],
        savename=savename
    )
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process parameters for data generation.")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level.")
    parser.add_argument("--Tmax", type=int, default=1000, help="Final time.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap with the GS.")
    parser.add_argument("--dt", type=float, default=1, help="Time step.")
    parser.add_argument("--tol", type=float, nargs='+', default=[0.1], help="SVD tolerances.")
    parser.add_argument("--step", type=int, default=1, help="Successive length.")
    parser.add_argument("--fudge_factor", type=float, default=1.0, help="Left and right endpoint fudge factor.")
    parser.add_argument("--option", type=str, default='', help="Which kind of energy casting.")
    parser.add_argument("--baseline", type=lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"], default=True, help="Baseline ODMD or not")
    parser.add_argument("--stacked", type=lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"], default=True, help="Stack noisy data and denoised data")
    parser.add_argument("--depolarized", type=float, default=0, help="Depolarization damping factor")
    
    args = parser.parse_args()

    main(args.molecule, args.noise, args.Tmax, args.overlap, args.dt, args.tol, args.step, args.fudge_factor, args.option, args.baseline, args.stacked, args.depolarized)
