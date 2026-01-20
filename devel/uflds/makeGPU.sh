
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_INFO=4

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
logfile="time1_gpu_26"
# taskset -c 0-15 ./time1 -logfile $logfile
nsys profile --trace=cuda,openmp --stats=true --cuda-memory-usage=true --output="${logfile}_report" --force-overwrite=true ./time1 -logfile $logfile".log" > "$logfile"_nvidia_output.log


# sqlite3 -header -column time1_gpu_26_report.sqlite "
# SELECT
#   sd.value AS kernel_name,
#   k.gridX, k.gridY, k.gridZ,
#   k.blockX, k.blockY, k.blockZ,
#   (k.gridX*k.gridY*k.gridZ) AS blocks_per_launch,
#   (k.blockX*k.blockY*k.blockZ) AS threads_per_block,
#   (k.gridX*k.gridY*k.gridZ*k.blockX*k.blockY*k.blockZ) AS threads_per_launch,
#   COUNT(*) AS launches
# FROM CUPTI_ACTIVITY_KIND_KERNEL k
# JOIN StringIds sd ON sd.id = k.demangledName
# GROUP BY kernel_name, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
# ORDER BY launches DESC, threads_per_launch DESC;
# "