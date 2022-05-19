#!/bin/sh

mkdir -p logs
for d in 'biosnap' 'bindingdb' 'davis' ;
do
    for i in 1 2 3 4 5 ;
    do
        echo "CUDA_VISIBLE_DEVICES=0,2,5,6 python train.py --task $d >> logs/log_${d}_${i}.txt"
        CUDA_VISIBLE_DEVICES=0,2,5,6 python train.py --task $d >> logs/log_${d}_${i}.txt
    done
done
