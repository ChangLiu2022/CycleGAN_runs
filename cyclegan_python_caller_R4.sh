#!/bin/bash

# Script to call the run-file for cycleGAN training (Run 5)

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python cgan_longterm_stab_compare.py --config RR_CGAN_SAVE_R4 --day0s "20201211,20210225,20211019,20211129,20230117,20230213"

echo "✅ Training and Evaluation completed!"