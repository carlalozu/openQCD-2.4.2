
#!/bin/bash
# run file env.txt
cd ../..
source env.txt

# copy plaq_sum_GPU into plaq_sum.c
cd modules/uflds
cp plaq_sum_GPU.c plaq_sum.c

cd ../../devel/uflds
make -f Makefile_GPU