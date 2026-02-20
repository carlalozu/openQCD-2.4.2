export LLVM_HOME=$SCRATCH/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

export MPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi"
export MPI_INCLUDE="/usr/lib/x86_64-linux-gnu/openmpi/include"
export OMPI_CC="$(which clang)"

ROOT=$SCRATCH/openQCD-2.4.2
DIR=$ROOT/scripts
file=time_gpu

cd $ROOT/devel/uflds
> $file.log

base_t1=64

perl -i -pe "s/#define L0 \\d+/#define L0 $base_t1/" $ROOT/include/global.h
perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD $base_t1/" $ROOT/include/global.h


make clean
make time
./time

cat time.log >> $file.log
mv $file.log $ROOT/output/$file.log


cat $ROOT/output/$file.log | python $DIR/parse.py > $ROOT/output/$file.csv
