# Code to accompany "From noisy observables to accurate ground‑state energies: a  quantum-classical signal subspace approach with denoising"
Hardeep Bassi, Yizhi Shen, Harish S. Bhat, and Roel Van Beeumen. Please direct any questions to `hbassi2@ucmerced.edu`.

## Generate the noisy and noiseless data 
Specify a `molecule`, `noise` parameter for the standard deviation of the statistical noise from finite shots, final time evolution `Tmax`, `overlap` with the ground state $p_0$, timestep `dt` $\Delta t$, `num_trajs` the amount of trajectories, and the depolarizing noise strength parameter `$\gamma$`. Set $\gamma$ = -1.0 if depolarizing noise is not desired. An example would be:

`python generate_data.py --molecule Cr2 --noise 0.1 --Tmax 1000 --overlap 0.2 --dt 1 --num_trajs 1 --gamma -1.0`

## Fourier denoising data generation
Specify the desired filepaths of the noisy data generated from `generate_data.py` to be denoised and the denoising thresholding parameters within the script and run:

`python fourier_denoising.py`

Following this, use `stack_data.ipynb` to assemble the stacked dataset from whichever denoising parameters are desired.

## Run ODMD  
Use the same parameters as used for data generation and set `baseline` to `True` or `False` depending on if baseline ODMD is desired, and `stacked` to `True` or `False` depending on if FDODMD is desired. An example would be:

`python run_odmd.py --molecule Cr2 --noise 0.1 --Tmax 1000 --overlap 0.2 --dt 1 --tol 0.8 0.1 0.01 0.001 --step 1 --fudge_factor 0.2 --option ff=0.2_left_right --baseline False --stacked True --depolarized -1.0`

## Zero-padding/FFT
Run `run_zeropadding.sh` from the command line and adjust the parameters as needed.

## All files above use `utils.py` for specific implementation of the algorithms presented in the paper

## Figures
Inside of the `figures` folder, `create_plots_FD.ipynb`, `create_plots_SE.ipynb`, and `heatmap.ipynb` can be used to re-create the Figures from the manuscript. All relevant data has been provided in the `data` and `figures` folders.
