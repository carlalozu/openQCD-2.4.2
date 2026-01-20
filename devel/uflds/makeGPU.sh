
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# run file env.txt
cd ../..
source env.txt

# copy plaq_sum_GPU into plaq_sum.c
cd modules/uflds
cp plaq_sum_GPU.c plaq_sum.c

# make
cd ../../devel/uflds
make -f Makefile_GPU clean
make -f Makefile_GPU

# run
logfile="time1_gpu_1.log"
taskset -c 0-15 ./time1 -logfile $logfile