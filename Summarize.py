from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "facebook/bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
import tarfile
from google.colab import drive
from datasets import load_dataset

drive.mount('/content/drive')
tar_path = "/content/drive/MyDrive/dataset/uzbek_XLSum_v2.0.tar.bz2"
extract_path = "/content/xlsum_uzbek"

with tarfile.open(tar_path, "r:bz2") as tar:
    tar.extractall(extract_path)

dataset = load_dataset("json", data_files={
    "train": "/content/xlsum_uzbek/uzbek_train.jsonl",
    "validation": "/content/xlsum_uzbek/uzbek_val.jsonl",
    "test": "/content/xlsum_uzbek/uzbek_test.jsonl"
})
def token(examples):
    model_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = tokenizer(
        text_target=examples["summary"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = dataset.map(token, batched=True,
                              remove_columns=dataset["train"].column_names)
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="/content/summarization_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=50,
    dataloader_num_workers=2,
    report_to="none"
)
!pip install evaluate rouge_score -q
import evaluate
import numpy as np

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels,
                           use_stemmer=False)
    return {"rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4)}
from transformers import Seq2SeqTrainer, EarlyStoppingCallback

model.gradient_checkpointing_enable()

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.save_model("/content/summarization_model/final")
tokenizer.save_pretrained("/content/summarization_model/final")
results = trainer.evaluate(tokenized_data["test"])
print(results)
import json

path = "/content/drive/MyDrive/Colab Notebooks/Untitled1.ipynb"

with open(path, "r") as f:
    nb = json.load(f)

# state key qo'shish
if "widgets" in nb.get("metadata", {}):
    for widget_id, widget_data in nb["metadata"]["widgets"].items():
        if "state" not in widget_data:
            widget_data["state"] = {}

with open(path, "w") as f:
    json.dump(nb, f, indent=2)

print("Tayyor!")