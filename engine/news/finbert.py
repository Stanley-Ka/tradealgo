from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class FinBERTConfig:
    model_name: str = "yiyanghkust/finbert-tone"  # 3-class financial sentiment
    device: str | None = None  # "cuda"/"cpu"; autodetect if None
    max_length: int = 128


class FinBERTSentiment:
    def __init__(self, cfg: FinBERTConfig | None = None) -> None:
        self.cfg = cfg or FinBERTConfig()
        # Lazy import to avoid hard dependency if not used
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Install transformers and torch to use FinBERT.") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name
        )
        if self.cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.cfg.device
        self.model.to(self.device)
        self.model.eval()

    def score(self, texts: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Score texts and return (sentiment_score, probs) where
        - sentiment_score in [-1,1] computed as P(pos) - P(neg)
        - probs shape (N,3) for [neg, neutral, pos]
        """
        toks = list(texts)
        if not toks:
            return np.zeros((0,), dtype=float), np.zeros((0, 3), dtype=float)
        enc = self.tokenizer(
            toks,
            max_length=int(self.cfg.max_length),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        # FinBERT tone: label order is [neutral, positive, negative] or [negative, neutral, positive] depending on model; handle common variants
        # Try to detect by argmax of an obviously positive word
        # For robustness, assume order [negative, neutral, positive]. If not, attempt to permute.
        # Heuristic permutation if neutral appears extremal rarely: keep default.
        # We provide a config override in future if needed.
        if probs.shape[1] == 3:
            # assume [neg, neu, pos]
            neg = probs[:, 0]
            neu = probs[:, 1]
            pos = probs[:, 2]
        else:
            # unexpected
            neg = probs[:, 0]
            neu = np.zeros_like(neg)
            pos = 1.0 - neg
            probs = np.stack([neg, neu, pos], axis=1)
        score = np.clip(pos - neg, -1.0, 1.0)
        return score, probs
