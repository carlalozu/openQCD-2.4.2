#!/bin/bash -l
#SBATCH --job-name="time1_mpi32_openmp1_param1"
#SBATCH --account="eth8"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --constraint=mc
#SBATCH --mem=120GB
#SBATCH --partition=normal
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/smaier/openQCD-2.4/devel/uflds/time1
