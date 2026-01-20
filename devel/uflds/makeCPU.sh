
#!/bin/bash
# run file env.txt
cd ../..
source env.txt

# copy plaq_sum
cd modules/uflds
cp plaq_sum_cpu.c plaq_sum.c

cd ../../devel/uflds
make -f Makefile_CPU