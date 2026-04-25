#!/bin/bash

#SBATCH --job-name=LSTM_EM_1    # create a short name for your job
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.log
#SBATCH --mail-user=eddyliu@umich.edu
#SBATCH --mail-type=END,FAILED
#SBATCH --account=cchestek0
#SBATCH --partition=spgpu
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus=a40:1                # number of gpus per node
#SBATCH --cpus-per-gpu=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-gpu=128G     # memory per gpu- allocate as needed
#SBATCH --time=300:00:00         # total run time limit (HH:MM:SS)

# Initialize conda for the shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate nd

./LSTM_EM_python_caller_1.sh