#!/bin/bash -l
#SBATCH --job-name="horizon_svf_EXTPAR"
#SBATCH --account="pr133"
#SBATCH --time=00:07:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread
#SBATCH --output=horizon_svf_EXTPAR.o
#SBATCH --error=horizon_svf_EXTPAR.e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate raytracing
srun -u python run_daint.py