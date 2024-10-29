#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --time=2

# Load MATLAB module
module load matlab

# Define arrays of parameters
molecules=("Cr2")
noises=(0.1 1.0 2.0)
overlaps=(0.2 0.72)
tols=("0.01, 0.1, 0.9999, 1.9999")
Tmaxs=(3000)
dts=(1)

# Iterate over parameter sets and submit jobs
for molecule in "${molecules[@]}"; do
  for noise in "${noises[@]}"; do
    for overlap in "${overlaps[@]}"; do
      for tol in "${tols[@]}"; do
        for Tmax in "${Tmaxs[@]}"; do
          for dt in "${dts[@]}"; do
            # Create a unique job name
            jobname="${molecule}_n${noise}_o${overlap}_Tmax${Tmax}_dt${dt}"

            # Construct the MATLAB command
            matlab_cmd="test('molecule', '${molecule}', 'noise', ${noise}, 'overlap', ${overlap}, 'tol', [${tol}], 'Tmax', ${Tmax}, 'dt', ${dt}); exit;"

            # Submit the job
            sbatch --job-name=${jobname} --output=${jobname}.out --error=${jobname}.err --wrap="srun -n 1 -c 32 matlab -nodisplay -r \"${matlab_cmd}\" -logfile ${jobname}.log"
          done
        done
      done
    done
  done
done
