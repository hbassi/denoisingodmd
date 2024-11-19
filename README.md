# denoisingodmd

## Generate the data via:
python generate_data.py --molecule Cr2 --noise 0.1 --Tmax 1000 --overlap 0.2 --dt 1 --num_trajs 1

## Run ODMD via:
python run_odmd.py --molecule Cr2 --noise 0.1 --Tmax 1000 --overlap 0.2 --dt 1 --tol 0.8 0.1 0.01 0.001 --step 1 --fudge_factor 0.2 --option ff=0.2_left_right --baseline True


## Fourier denoising (for now):
in the notebook fourier_denoising.ipynb (TODO: port to script). After saving the denoised trajectory from the notebook, run the ODMD command above with the flag --baseline False

## Zero-padding/FFT

Run run_zeropadding.sh from the command line and adjust internal parameters as needed. If multiplicative padding, modify line 44 in utils to be n * numpad. If additive padding, leave as is (TODO: fix this to not be hard coded)
