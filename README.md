# Multilingual Morphological Inflection Generation for Low-Resource Languages
This repository contains the code and data for my bachelor's thesis.
## Models
- `./ByT5`: Experiments with ByT5
- `./llm`: Zero-Shot prompting with GPT-OSS-120B and Gemini 2.5 Flash

## Data 
- `./2023InflectionST/part1/data`: 2023 Shared task data + adding updated data from previous UniMorph releases, in triplet format
- `./preprocessing`: Data preprocessing (contains fine-tuning and pre-training data for ByT5)
- `./postprocessing`: Data post-processing (used for evaluation on mixed datasets)
