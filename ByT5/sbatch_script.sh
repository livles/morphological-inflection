#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:4
#SBATCH --time 2-00:00:00
#SBATCH --output %j.out

python3 fine_tuned_ByT5.py

