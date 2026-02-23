export LLVM_HOME=$SCRATCH/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

export MPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi"
export MPI_INCLUDE="/usr/lib/x86_64-linux-gnu/openmpi/include"
export OMPI_CC="$(which clang)"

export GOMP_CPU_AFFINITY=0-16

ROOT=$SCRATCH/openQCD-2.4.2
DIR=$ROOT/scripts
file=time_threads

cd $ROOT/devel/uflds
> $file.log

perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 8/" $ROOT/include/global.h

perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 8/" $ROOT/include/global.h

for i in 8 16 32 64 128
do
    base_t1=$((4 * i))
    echo $base_t1
    perl -i -pe "s/#define L0 \\d+/#define L0 $base_t1/" $ROOT/include/global.h

    for t in 1 2 4 8 16
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