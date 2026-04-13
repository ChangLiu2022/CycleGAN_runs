#!/bin/bash

# Script to call the run-file for cycleGAN training

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python cgan_longterm_stab_compare.py --config RR_CGAN_SAVE_R1 --day0s "20200302,20210525,20210730,20220407,20220817,20230622"

echo "✅ Training and Evaluation completed!"