
ROOT=$SCRATCH/openQCD-2.4.2

#  with no threading 
perl -i -pe "s/#define NPROC0 \\d+/#define NPROC0 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC1 \\d+/#define NPROC1 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC2 \\d+/#define NPROC2 1/" $ROOT/include/global.h
perl -i -pe "s/#define NPROC3 \\d+/#define NPROC3 1/" $ROOT/include/global.h

perl -i -pe "s/#define L0 \\d+/#define L0 8/" $ROOT/include/global.h
perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 8/" $ROOT/include/global.h

perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 4/" $ROOT/include/global.h

> profile.log

echo "" > profile.log
echo "CPU profile" > profile.log
make -f Makefile.cpu clean
make -f Makefile.cpu -j || exit 1

./time_plaq_dble >> profile.log
./time_plaq_sum >> profile.log

echo "" >> profile.log
echo "GPU profile" >> profile.log
make -f Makefile.gpu clean
make -f Makefile.gpu -j || exit 1

./time_plaq_dble >> profile.log
./time_plaq_dblev >> profile.log
./time_plaq_sum >> profile.log
./time_plaq_sumv >> profile.log

echo "" >> profile.log
echo "GPU profile lex layout" >> profile.log
make -f Makefile.gpu-lex clean
make -f Makefile.gpu-lex -j || exit 1

./time_plaq_dble >> profile.log
./time_plaq_dblev >> profile.log
./time_plaq_sum >> profile.log
./time_plaq_sumv >> profile.log