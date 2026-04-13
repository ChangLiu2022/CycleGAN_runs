#!/bin/bash

# Script to call the run-file for cycleGAN training (Test Run)

echo "🚀 Starting CycleGAN Model Training and Evaluation (Rtest)..."

python cgan_longterm_stab_compare.py --config RR_CGAN_SAVE_Rtest --day0s "20230621,20230622"

echo "✅ Training and Evaluation completed!"
