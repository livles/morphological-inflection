
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
      

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(10))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(10))

dataset = dataset.map(tokenize, batched=True)


from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-small",
    dtype=torch.float16,
    device_map="auto"
)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    eval_strategy="epoch",
    push_to_hub=True,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

