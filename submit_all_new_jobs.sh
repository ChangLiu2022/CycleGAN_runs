#!/bin/bash

echo "Submitting new LSTM training jobs to Slurm..."

# Submit EM runs
echo "Submitting EM 2 and 3..."
sbatch LSTM_EM_batch_script_2.sh
sbatch LSTM_EM_batch_script_3.sh

# Submit NN run
echo "Submitting NN 2..."
sbatch LSTM_NN_batch_script_2.sh

# Submit NO run
echo "Submitting NO 2..."
sbatch LSTM_NO_batch_script_2.sh

echo "✅ All new jobs have been submitted to the queue!"
