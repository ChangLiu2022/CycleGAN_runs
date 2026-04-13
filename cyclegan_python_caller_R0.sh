#!/bin/bash

# Script to call the run-file for cycleGAN training

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python cgan_longterm_stab_compare.py --config RR_CGAN_SAVE_R0 --day0s "20200127,20210628,20210713,20220428,20220718"


echo "✅ Training and Evaluation completed!"
