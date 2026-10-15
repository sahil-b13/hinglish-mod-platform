"""
Fine-tunes HingBERT/HingRoBERTa on the Bohra 2018 dataset (or any CSV with
`text`,`label` columns).

**You run this — I can't, my sandbox has no GPU and no Hugging Face access.**
Recommended: upload this whole repo to Google Colab or Kaggle, attach a free
GPU runtime, then run:

    python scripts/train.py \
        --data_dir data/bohra \
        --model_name l3cube-pune/hing-roberta \
        --output_dir models/hingroberta-finetuned \
        --epochs 4

Expects:
    data/bohra/train.csv   (columns: text, label)
    data/bohra/test.csv    (columns: text, label)

Labels should already be encoded as ints matching consumer/classifier.py's
id2label mapping: 0=safe, 1=abusive, 2=severe. Adjust NUM_LABELS / id2label below
if your label scheme differs (e.g. binary hate/not-hate).

After training, evaluate with:
    python scripts/evaluate.py --model_path models/hingroberta-finetuned --test_csv data/bohra/test.csv
(stub also provided in this scripts/ folder)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

NUM_LABELS = 3
ID2LABEL = {0: "safe", 1: "abusive", 2: "severe"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_split(data_dir: Path, split: str):
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See README §2 for how to get Bohra data, "
            f"and format it as text,label CSVs before running this script."
        )
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), f"{path} must have 'text' and 'label' columns"
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(ID2LABEL.keys())
    )
    metrics = {"macro_f1": macro_f1}
    for i, label_name in ID2LABEL.items():
        metrics[f"f1_{label_name}"] = f1_per_class[i]
    return metrics


def main(args):
    # Imports deferred so `python scripts/train.py --help` doesn't require torch installed
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    data_dir = Path(args.data_dir)
    train_df = load_split(data_dir, "train")
    test_df = load_split(data_dir, "test")

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print("Label distribution (train):")
    print(train_df["label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training. Watch the eval_macro_f1 each epoch — "
          "plan target is ~0.68-0.73 on Bohra (per literature). "
          "Stop/adjust epochs if it plateaus or starts overfitting (train loss "
          "keeps dropping but eval_macro_f1 stalls/drops).")
    trainer.train()

    final_metrics = trainer.evaluate()
    print("Final eval metrics:", final_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")
    print("Next: point consumer/classifier.py's get_classifier() at this path "
          "(see the TODO comment in that file), then run HateCheckHIn diagnostics "
          "per README/plan §5 Week 3-4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune HingBERT/HingRoBERTa on Bohra data")
    parser.add_argument("--data_dir", required=True, help="Dir containing train.csv / test.csv")
    parser.add_argument("--model_name", default="l3cube-pune/hing-roberta",
                         help="HF model id, e.g. l3cube-pune/hing-roberta or l3cube-pune/hing-bert")
    parser.add_argument("--output_dir", default="models/hingroberta-finetuned")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    main(args)
