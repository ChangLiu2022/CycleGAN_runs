#!/bin/bash

# Script to call the run-file for cycleGAN training

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python LSTM_LS_LINK_START.py --config LSTM_NN_LM_1 --starting_date_idx 130

echo "✅ Training and Evaluation completed!"
