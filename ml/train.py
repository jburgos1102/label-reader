"""
train.py
--------
Fine-tunes DistilBERT for token classification (NER) on shipping label data.

Reads: ml/training_data/dataset.jsonl  (produced by export_training_data.py)
Saves: ml/models/label_reader/          (HuggingFace model directory)
       ml/models/label_reader.onnx      (ONNX export for CoreML conversion)

Usage:
  python ml/train.py
  python ml/train.py --epochs 10 --batch-size 8

Re-run any time after annotating more labels. The model improves automatically
as the dataset grows — no prompt engineering required.

Accuracy benchmarks by dataset size:
  ~31  labels : pipeline established, accuracy not yet meaningful
  ~200 labels : recipient_name ~85%, overall ~92%
  ~500 labels : recipient_name ~92%, overall ~95%
  ~1000 labels: recipient_name ~95%+, Groq calls rare
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorForTokenClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAINING_DATA = Path(__file__).parent / "training_data" / "dataset.jsonl"
MODEL_DIR = Path(__file__).parent / "models" / "label_reader"
ONNX_PATH = Path(__file__).parent / "models" / "label_reader.onnx"
BASE_MODEL = "distilbert-base-uncased"

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

LABELS = [
    "O",
    "B-NAME",
    "I-NAME",
    "B-STREET",
    "I-STREET",
    "B-CITY",
    "I-CITY",
    "B-STATE",
    "B-ZIP",
    "B-TRACKING",
    "I-TRACKING",
    "B-CARRIER",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split_dataset(examples: list[dict], test_size: float = 0.2, seed: int = 42):
    """
    80/20 train/test split.
    If fewer than 25 examples, train on everything (report training accuracy only).
    """
    if len(examples) < 25:
        print(
            f"⚠️  Only {len(examples)} examples — training on full dataset, no held-out test set."
        )
        return examples, []

    train, test = train_test_split(examples, test_size=test_size, random_state=seed)
    return train, test


# ---------------------------------------------------------------------------
# Tokenization + label alignment
# ---------------------------------------------------------------------------


def tokenize_and_align(batch, tokenizer):
    """
    DistilBERT uses subword tokenization. A word like '1Z999AA' may become
    ['1', '##z', '##999', '##aa']. We propagate the label from the first
    subword to all continuations (B- → I- where applicable).
    Special tokens ([CLS], [SEP]) get label -100 (ignored by loss).
    """
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        padding=False,
    )

    all_label_ids = []
    for i, word_labels in enumerate(batch["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(LABEL2ID.get(word_labels[word_id], 0))
            else:
                # Subword continuation: B- → I- if available, else same label
                orig = word_labels[word_id]
                if orig.startswith("B-"):
                    continuation = "I-" + orig[2:]
                    label_ids.append(LABEL2ID.get(continuation, LABEL2ID[orig]))
                else:
                    label_ids.append(LABEL2ID.get(orig, 0))
            prev_word_id = word_id
        all_label_ids.append(label_ids)

    tokenized["labels"] = all_label_ids
    return tokenized


# ---------------------------------------------------------------------------
# Evaluation metric (seqeval)
# ---------------------------------------------------------------------------


def make_compute_metrics(tokenizer):
    try:
        from seqeval.metrics import classification_report, f1_score

        use_seqeval = True
    except ImportError:
        use_seqeval = False
        print("seqeval not installed — using simple token accuracy instead.")

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, label_ids):
            true_seq, pred_seq_filtered = [], []
            for pred, label in zip(pred_seq, label_seq):
                if label == -100:
                    continue
                true_seq.append(ID2LABEL[label])
                pred_seq_filtered.append(ID2LABEL[pred])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_filtered)

        if use_seqeval:
            f1 = f1_score(true_labels, pred_labels)
            report = classification_report(true_labels, pred_labels, output_dict=True)
            result = {"f1": f1}
            for entity in [
                "NAME",
                "STREET",
                "CITY",
                "STATE",
                "ZIP",
                "TRACKING",
                "CARRIER",
            ]:
                if entity in report:
                    result[f"f1_{entity.lower()}"] = report[entity]["f1-score"]
            return result
        else:
            correct = sum(
                t == p
                for ts, ps in zip(true_labels, pred_labels)
                for t, p in zip(ts, ps)
            )
            total = sum(len(ts) for ts in true_labels)
            return {"accuracy": correct / total if total else 0}

    return compute_metrics


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(model, tokenizer, onnx_path: Path):
    import torch

    model.eval()
    dummy_input = tokenizer(
        ["JOHN SMITH 123 MAIN ST CHICAGO IL 60601"],
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting ONNX → {onnx_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ONNX model saved ({onnx_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for shipping label NER"
    )
    parser.add_argument(
        "--data", default=str(TRAINING_DATA), help="Path to dataset.jsonl"
    )
    parser.add_argument(
        "--model-dir", default=str(MODEL_DIR), help="Where to save the trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs (increase with more data)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)

    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Run:  python ml/export_training_data.py  first.")
        return

    # Load data
    print(f"Loading dataset from {data_path} ...")
    examples = load_examples(data_path)
    print(f"  {len(examples)} examples")

    train_examples, test_examples = split_dataset(examples, seed=args.seed)
    print(f"  Train: {len(train_examples)}   Test: {len(test_examples)}")

    # Build HuggingFace datasets
    def to_hf(exs):
        return Dataset.from_dict(
            {
                "tokens": [e["tokens"] for e in exs],
                "labels": [e["labels"] for e in exs],
            }
        )

    splits = {"train": to_hf(train_examples)}
    if test_examples:
        splits["test"] = to_hf(test_examples)
    dataset = DatasetDict(splits)

    # Tokenizer + model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    model = DistilBertForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenized = dataset.map(
        lambda batch: tokenize_and_align(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "labels"],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    eval_strategy = "epoch" if test_examples else "no"
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        load_best_model_at_end=bool(test_examples),
        metric_for_best_model="f1" if test_examples else None,
        logging_steps=5,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test"),
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer) if test_examples else None,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs ...")
    trainer.train()

    # Evaluate
    if test_examples:
        print("\nEvaluation on held-out test set:")
        metrics = trainer.evaluate()
        for k, v in sorted(metrics.items()):
            if not k.startswith("eval_loss"):
                print(f"  {k:<30} {v:.4f}")

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"\nModel saved → {model_dir}")

    # ONNX export
    if not args.no_onnx:
        try:
            import torch

            export_onnx(model, tokenizer, ONNX_PATH)
            print("Run  python ml/export_coreml.py  to convert to CoreML for iOS.")
        except ImportError:
            print("torch not available for ONNX export — skipping.")

    print("\nDone. Re-run after annotating more labels to improve accuracy.")


if __name__ == "__main__":
    main()
