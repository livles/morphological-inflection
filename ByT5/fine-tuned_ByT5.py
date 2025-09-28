# source: https://learnopencv.com/fine-tuning-t5/ https://huggingface.co/docs/transformers/v4.28.1/tasks/summarization
from huggingface_hub import login

login()

import torch

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset

MODEL = 'google/byt5-small'
BATCH_SIZE = 48
NUM_PROCS = 16
EPOCHS = 3
OUT_DIR = "results_byt5-small"
MAX_LENGTH = 256 # Maximum context length to consider while preparing dataset.

dataset_train = load_dataset(
    'csv',
    data_files='input/train.csv',
    split='train'
)
dataset_valid = load_dataset(
    'csv',
    data_files='input/valid.csv',
    split='train'
)

dataset_train = dataset_train.shuffle(seed=42).select(range(100))
dataset_valid = dataset_valid.shuffle(seed=42).select(range(100))
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Function to convert text data into model inputs and targets
def preprocess_function(examples):
    inputs = [f"assign tag: {title} {body}" for (title, body) in zip(examples['Title'], examples['Body'])]
    # Set up the tokenizer for targets
    cleaned_tag = [' '.join(''.join(tag.split('<')).split('>')[:-1]) for tag in examples['Tags']]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )

    labels = tokenizer(
        text_target=cleaned_tag,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the function to the whole dataset
tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)
tokenized_valid = dataset_valid.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-small",
    dtype=torch.float16,
    device_map="auto"
)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

import evaluate
import numpy as np
rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    weight_decay=0.01,

    save_total_limit=3,

    num_train_epochs=4,

    predict_with_generate=True,

    fp16=False,

    push_to_hub=True,
)



trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_valid,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)



history = trainer.train()
trainer.push_to_hub()
