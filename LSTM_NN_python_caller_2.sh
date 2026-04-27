#!/bin/bash

# Script to call the run-file for cycleGAN training

echo "🚀 Starting CycleGAN Model Training and Evaluation..."

python LSTM_LS_LINK_START.py --config LSTM_NN_LM_2 --starting_date_idx 180

echo "✅ Training and Evaluation completed!"
