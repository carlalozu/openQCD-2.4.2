
#!/bin/bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# run file env.txt
cd ../..
source env.txt

# copy plaq_sum_cpu.c into plaq_sum.c
cd modules/uflds
cp plaq_sum_cpu.c plaq_sum.c

# make
cd ../../devel/uflds
make -f Makefile_CPU clean
make -f Makefile_CPU

# use only one node
logfile="time1_cpu_1.log"
taskset -c 0-15 ./time1 -logfile $logfile