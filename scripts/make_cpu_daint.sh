#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --account=c37
#SBATCH --environment=cpe-cray-24.07
#SBATCH --exclusive

# CC=cc
# CFLAGS = -std=c11 -pedantic -fstrict-aliasing -fopenmp -Wall \
#          -Wno-long-long -Wno-format-overflow -Wstrict-prototypes \
#          -O  -DFMA3 -DPM
         
export GOMP_CPU_AFFINITY=0-71

ROOT=$SCRATCH/openQCD-2.4.2
DIR=$ROOT/scripts
file=time_threads_daint

cd $ROOT/devel/uflds
> $file.log

perl -i -pe "s/#define L1 \\d+/#define L1 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 4/" $ROOT/include/global.h

perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 4/" $ROOT/include/global.h

for i in 8 16 32 64 128
do
    base_t1=$((8 * 4 * i))
    echo $base_t1
    perl -i -pe "s/#define L0 \\d+/#define L0 $base_t1/" $ROOT/include/global.h

    for t in 1 2 4 8 16 32 64 72
    do
        thread_t1=$((base_t1 / t))
        perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD $thread_t1/" $ROOT/include/global.h
        make clean
        make time
        ./time

        cat time.log >> $file.log
    done

done

mv $file.log $ROOT/output/$file.log

cat $ROOT/output/$file.log | python $DIR/parse.py > $ROOT/output/$file.csv
python $DIR/plot_roofline_cpu.py 