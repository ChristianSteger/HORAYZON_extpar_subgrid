#!/bin/bash

#SBATCH --job-name="refine_mesh"
#SBATCH --nodes=1
#SBATCH --output="job.out"
#SBATCH --time=00:17:00
#SBATCH --partition=postproc
#SBATCH --cpus-per-task=64
#SBATCH --exclusive

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# conda activate horayzon_extpar_subgrid
# srun -u python 1_refine_icon_mesh.py
/users/csteger/miniconda3/envs/horayzon_extpar_subgrid/bin/python -u 1_refine_icon_mesh.py
