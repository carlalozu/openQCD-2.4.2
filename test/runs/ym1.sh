#!/bin/bash -l
#SBATCH --job-name="ym1_prof_run1"
#SBATCH --account="eth8"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=mc
#SBATCH --mem=120GB
#SBATCH --partition=normal
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/smaier/openQCD-2.4/main/ym1 -i /users/smaier/openQCD-2.4/main/ym1.in -noms
