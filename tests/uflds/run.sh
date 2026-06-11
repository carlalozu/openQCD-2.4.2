# Run tests with no threading and threading on the CPU
export OMPI_CC=/opt/homebrew/opt/llvm/bin/clang
export OMPI_CXX=/opt/homebrew/opt/llvm/bin/clang++
export GCC=/opt/homebrew/opt/llvm/bin/clang
export MPI_INCLUDE=/opt/homebrew/Cellar/open-mpi/5.0.9/include
export PATH="/opt/homebrew/bin:$PATH"
export OMP_INCLUDE=/opt/homebrew/Cellar/libomp/22.1.2/include


ROOT=$SCRATCH/openQCD-2.4.2

#  with no threading 
perl -i -pe "s/#define NPROC0 \\d+/#define NPROC0 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC1 \\d+/#define NPROC1 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC2 \\d+/#define NPROC2 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC3 \\d+/#define NPROC3 1/" $ROOT/include/global.h

perl -i -pe "s/#define L0 \\d+/#define L0 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1 \\d+/#define L1 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 4/" $ROOT/include/global.h

perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 4/" $ROOT/include/global.h

# CPU tests
make -f Makefile.cpu clean
make -f Makefile.cpu -j

./check4
./check5


#  with 4 OpenMP threads
perl -i -pe "s/#define L0 \\d+/#define L0 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 4/" $ROOT/include/global.h


# CPU tests
make -f Makefile.cpu clean
make -f Makefile.cpu -j

./check4
./check5