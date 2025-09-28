#!/bin/bash
# SBATCH --partition lrz-hgx-h100-94x4
# SBATCH --gres gpu:1
# SBATCH --time 2-00:00:00
# SBATCH --dependency afterok:<prev_jobid>
# SBATCH --output %j.out

!python3 fine-tuned_ByT5.py

!wget "https://www.dropbox.com/scl/fi/525gv6tmdi3n32mipo6mr/input.zip?rlkey=5jdsxahphk2ped5wxbxnv0n4y&dl=1" -O input.zip
!ls
!unzip input.zip

!pip install transformers torch datasets evaluate rouge_score pip accelerate
!pip install transformers --upgrade


!nvidia-smi
!pip install ipywidgets

# +
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/byt5-small"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "livles/results_byt5-small",
    dtype=torch.float16,
    device_map="auto"
)

input_ids = tokenizer("assign tag{title} {body}: ('Photosynthesis','Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy.)''", return_tensors="pt").to(model.device)

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# -


