
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

> test.log

make -f Makefile.cpu clean
make -f Makefile.cpu -j || exit 1

echo "CPU tests" >> test.log
./check4 >> test.log

make -f Makefile.gpu clean
make -f Makefile.gpu -j || exit 1

echo "GPU tests" >> test.log
./check4 >> test.log
./check5 >> test.log

make -f Makefile.gpu-lex clean
make -f Makefile.gpu-lex -j || exit 1

echo "GPU tests lex" >> test.log
./check4-lex >> test.log
./check5 >> test.log

#  with 4 OpenMP threads
perl -i -pe "s/#define L0 \\d+/#define L0 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 4/" $ROOT/include/global.h


make -f Makefile.cpu clean
make -f Makefile.cpu -j || exit 1

echo "CPU tests" >> test.log
./check4 >> test.log

make -f Makefile.gpu clean
make -f Makefile.gpu -j || exit 1

echo "GPU tests" >> test.log
./check4 >> test.log
./check5 >> test.log

make -f Makefile.gpu-lex clean
make -f Makefile.gpu-lex -j || exit 1

echo "GPU tests lex" >> test.log
./check4-lex >> test.log
./check5 >> test.log