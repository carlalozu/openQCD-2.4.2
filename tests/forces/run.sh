export OMPI_CC=nvc
export MPICH_CC=nvc
export CC=mpicc

ROOT=$SCRATCH/openQCD-2.4.2

#  with no threading 
perl -i -pe "s/#define NPROC0 \\d+/#define NPROC0 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC1 \\d+/#define NPROC1 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC2 \\d+/#define NPROC2 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC3 \\d+/#define NPROC3 1/" $ROOT/include/global.h

perl -i -pe "s/#define L0 \\d+/#define L0 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 4/" $ROOT/include/global.h

perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 4/" $ROOT/include/global.h

name = "test_nvc"
> $name.log
# CPU tests
echo "CPU tests" > $name.log
make -f Makefile.cpu clean
make -f Makefile.cpu -j || exit 1

./check3 >> $name.log

# GPU tests
echo "GPU tests" >> $name.log
make -f Makefile.gpu clean
make -f Makefile.gpu -j || exit 1

./check3 >> $name.log

# GPU tests lex layout
echo "GPU tests lex layout" >> $name.log
make -f Makefile.gpu-lex clean
make -f Makefile.gpu-lex -j || exit 1

./check3-lex >> $name.log

#  with 4 OpenMP threads
perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 4/" $ROOT/include/global.h


echo "CPU tests" >> $name.log
make -f Makefile.cpu clean
make -f Makefile.cpu -j || exit 1

./check3 >> $name.log

echo "GPU tests" >> $name.log
make -f Makefile.gpu clean
make -f Makefile.gpu -j || exit 1

./check3 >> $name.log

# GPU tests lex layout
echo "GPU tests lex layout" >> $name.log
make -f Makefile.gpu-lex clean
make -f Makefile.gpu-lex -j || exit 1

./check3-lex >> $name.log
