#!/bin/bash
#SBATCH --partition lrz-hgx-a100-80x4
#SBATCH --gres gpu:1
#SBATCH --time 12:00:00
#SBATCH --output lrz-hgx-a100-80x4.out

python3 fine_tuned_ByT5.py

