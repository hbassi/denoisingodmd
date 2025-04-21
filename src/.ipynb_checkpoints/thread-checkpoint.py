import subprocess
from multiprocessing import Pool

# Define the base command and parameters
base_command = "python run_odmd.py"
common_args = "--molecule Cr2 --Tmax 1500 --overlap 0.2 --dt 1.0 --tol 0.8 0.5 0.1 0.01 --step 1 --fudge_factor 0.2 --option ff=0.2_left_right --baseline True --stacked False"
noise_levels = [0.1]  # List of noise levels

# Function to execute the script for a given noise level
def run_script(noise):
    command = f"{base_command} {common_args} --noise {noise}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True)  # Runs the script as a shell command

# Use multiprocessing to parallelize
if __name__ == "__main__":
    with Pool(len(noise_levels)) as pool:  # Use one process per noise level
        pool.map(run_script, noise_levels)
