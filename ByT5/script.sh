#!/bin/bash
# SBATCH --partition lrz-hgx-h100-94x4
# SBATCH --gres gpu:1
# SBATCH --time 2-00:00:00
# SBATCH --dependency afterok:<prev_jobid>
# SBATCH --output %j.out

!pip install transformers torch datasets evaluate rouge_score pip accelerate
!pip install transformers --upgrade
#!nvidia-smi
!pip install ipywidgets


!python3 fine-tuned_ByT5.py

!python3 test_fine-tuned_ByT5.py

