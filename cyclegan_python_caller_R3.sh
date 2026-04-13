#!/bin/bash

# Script to call the run-file for cycleGAN training (Run 4)

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python cgan_longterm_stab_compare.py --config RR_CGAN_SAVE_R3 --day0s "20201007,20210323,20210908,20220131,20220907,20230508"

echo "✅ Training and Evaluation completed!"