"""Fine-tune a FinBERT classifier on financial news.

Input CSVs require columns:
  - text: the headline/body text
  - label: either in {negative,neutral,positive} or {-1,0,1}

Example:
  python -m engine.news.finbert_finetune \
    --train data/news/train.csv --eval data/news/valid.csv \
    --base-model yiyanghkust/finbert-tone --out-dir models/finbert-finetuned \
    --epochs 3 --batch 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune FinBERT for financial sentiment")
    p.add_argument("--train", required=True, help="Train CSV with text,label")
    p.add_argument("--eval", type=str, default="", help="Eval CSV with text,label")
    p.add_argument("--base-model", type=str, default="yiyanghkust/finbert-tone")
    p.add_argument("--out-dir", type=str, default="models/finbert-finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    return p.parse_args(argv)


def _encode_labels(series: pd.Series) -> tuple[np.ndarray, dict]:
    # Accept {-1,0,1} or {negative,neutral,positive}
    s = series.astype(str).str.lower()
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    if set(s.unique()) <= set(mapping.keys()):
        y = s.map(mapping).astype(int).values
        return y, mapping
    # try numeric
    sn = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    # Map -1->0, 0->1, 1->2
    y = (sn + 1).clip(0, 2).values
    return y, mapping


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        from transformers import (  # type: ignore
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset  # type: ignore
        import evaluate  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Install transformers, datasets, evaluate for fine-tuning."
        ) from e

    os.makedirs(args.out_dir, exist_ok=True)
    tr = pd.read_csv(args.train)
    if "text" not in tr.columns or "label" not in tr.columns:
        raise RuntimeError("train CSV must have columns: text,label")
    y_tr, label_map = _encode_labels(tr["label"])  # 0=neg,1=neu,2=pos
    tr["label_id"] = y_tr
    ds_tr = Dataset.from_pandas(tr[["text", "label_id"]])

    ds_ev = None
    if args.eval:
        ev = pd.read_csv(args.eval)
        if "text" not in ev.columns or "label" not in ev.columns:
            raise RuntimeError("eval CSV must have columns: text,label")
        y_ev, _ = _encode_labels(ev["label"])
        ev["label_id"] = y_ev
        ds_ev = Dataset.from_pandas(ev[["text", "label_id"]])

    tok = AutoTokenizer.from_pretrained(args.base_model)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3
    )

    def _tok_fn(batch):
        return tok(batch["text"], truncation=True, padding=True, max_length=128)

    ds_tr = ds_tr.map(_tok_fn, batched=True)
    if ds_ev is not None:
        ds_ev = ds_ev.map(_tok_fn, batched=True)

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1_macro": metric_f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
        }

    args_tr = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.batch),
        per_device_eval_batch_size=int(args.batch),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        evaluation_strategy=("steps" if ds_ev is not None else "no"),
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=bool(ds_ev is not None),
        metric_for_best_model="f1_macro",
    )
    trainer = Trainer(
        model=mdl,
        args=args_tr,
        train_dataset=ds_tr,
        eval_dataset=ds_ev,
        compute_metrics=compute_metrics if ds_ev is not None else None,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[finetune] saved model -> {args.out_dir}")


if __name__ == "__main__":
    main()
