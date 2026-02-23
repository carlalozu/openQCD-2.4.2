
ROOT=$SCRATCH/openQCD-2.4.2
DIR=$ROOT/scripts
file=time_gpu_daint

cd $ROOT/devel/uflds
> $file.log

perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 8/" $ROOT/include/global.h

perl -i -pe "s/#define L1_TRD \\d+/#define L1_TRD 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2_TRD \\d+/#define L2_TRD 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3_TRD \\d+/#define L3_TRD 8/" $ROOT/include/global.h

for i in 4 8 16 32 64
do
    base_t1=$((4 * i))
    echo $base_t1
    perl -i -pe "s/#define L0 \\d+/#define L0 $base_t1/" $ROOT/include/global.h
    perl -i -pe "s/#define L0_TRD \\d+/#define L0_TRD $base_t1/" $ROOT/include/global.h


    make clean
    make time
    ./time

    cat time.log >> $file.log

done

mv $file.log $ROOT/output/$file.log

cat $ROOT/output/$file.log | python $DIR/parse.py > $ROOT/output/$file.csv
python $DIR/plot_roofline_gpu_geno.py 