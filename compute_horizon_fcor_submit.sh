#!/bin/bash

#SBATCH --job-name="comp_fcor"
#SBATCH --nodes=1
#SBATCH --output="job.out"
#SBATCH --time=00:57:00
#SBATCH --partition=postproc
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# conda activate horayzon_extpar_subgrid
# srun -u python compute_horizon_fcor.py
/users/csteger/miniconda3/envs/horayzon_extpar_subgrid/bin/python -u compute_horizon_fcor.py
