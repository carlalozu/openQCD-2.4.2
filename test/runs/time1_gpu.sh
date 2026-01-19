#!/bin/bash -l
#SBATCH --job-name="time1_mpi1_openmp12_gpu_parallel"
#SBATCH --account="c21"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/smaier/openQCD-2.4/devel/uflds/time1