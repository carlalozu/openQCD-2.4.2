#!/bin/bash -l
#SBATCH --job-name="check6_mpi1_openmp12_cpu_matrix"
#SBATCH --account="eth8"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/smaier/openQCD-2.4/devel/uflds/check6
