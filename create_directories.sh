#!/bin/bash

echo "Creating missing directories for logs and results..."

# Directory for SBATCH log files
mkdir -p logs

# Directories for CycleGAN outputs and checkpoints for all runs
mkdir -p results/cgan_r0/checkpoints
mkdir -p results/cgan_r0/kl
mkdir -p results/cgan_r1/checkpoints
mkdir -p results/cgan_r1/kl
mkdir -p results/cgan_r2/checkpoints
mkdir -p results/cgan_r2/kl
mkdir -p results/cgan_r3/checkpoints
mkdir -p results/cgan_r3/kl
mkdir -p results/cgan_r4/checkpoints
mkdir -p results/cgan_r4/kl
mkdir -p results/cgan_rtest/checkpoints
mkdir -p results/cgan_rtest/kl

# Directory for saving/loading data normalizers
mkdir -p results/normalizers

# Directory for Ridge Regression models
mkdir -p results/models/rr

# Other directories present in the workspace
mkdir -p results/alignment_models

echo "✅ All required directories have been created successfully!"
