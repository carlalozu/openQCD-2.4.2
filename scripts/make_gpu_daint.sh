#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=c37
#SBATCH --environment=cpe-cray-24.07
#SBATCH --exclusive

ROOT=$SCRATCH/openQCD-2.4.2
cd $ROOT/scripts

chmod +x time_run_gpu.sh
./time_run_gpu.sh