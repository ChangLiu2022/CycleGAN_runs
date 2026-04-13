#!/bin/bash

# Script to call the run-file for cycleGAN training (Run 3)

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python cgan_longterm_stab_compare.py --config RR_CGAN_SAVE_R2 --day0s "20200729,20210420,20210818,20220314,20221010,20230404"

echo "✅ Training and Evaluation completed!"