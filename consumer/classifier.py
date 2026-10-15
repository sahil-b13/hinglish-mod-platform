"""
Abuse classifier stage.

Two implementations:

1. RuleBasedClassifier — a keyword/heuristic placeholder. Works with zero setup so
   you can see the full pipeline run today. NOT a real classifier — don't report
   numbers from this, it's a stand-in.

2. HingBertClassifier — loads a fine-tuned HingBERT/HingRoBERTa checkpoint via
   Hugging Face `transformers`. This is what you swap in once you've run
   scripts/train.py on real data (see README §2-3). It is currently commented out
   at the bottom of this file with the exact code you'll need — just uncomment
   and point MODEL_PATH at your checkpoint.

To swap: change `get_classifier()` at the bottom to return HingBertClassifier(...)
instead of RuleBasedClassifier(). No other file needs to change — consumer.py just
calls classifier.predict(text) and expects {"label": str, "confidence": float}.
"""

import re
from typing import Dict


# Small hand-picked lexicon for the placeholder. Real severity/abuse detection
# should come from the fine-tuned model, not this list — this exists purely so
# the pipeline has *something* to route on before that model exists.
ABUSIVE_TERMS = {
    "bakwas", "bewakoof", "ghatiya", "chutiya", "kharab dimag",
    "gaddar", "barbaad", "nikamma", "faltu",
}
SEVERE_TERMS = {
    "chutiya", "gaddar", "barbaad kar denge", "maar denge",
}


class RuleBasedClassifier:
    """Placeholder scorer: keyword hits + simple heuristics -> pseudo-probability."""

    def predict(self, text: str) -> Dict:
        t = text.lower()
        severe_hits = sum(1 for term in SEVERE_TERMS if term in t)
        abusive_hits = sum(1 for term in ABUSIVE_TERMS if term in t)

        if severe_hits > 0:
            label = "severe"
            confidence = min(0.6 + 0.15 * severe_hits, 0.97)
        elif abusive_hits > 0:
            label = "abusive"
            confidence = min(0.55 + 0.1 * abusive_hits, 0.9)
        else:
            label = "safe"
            confidence = 0.85

        return {"label": label, "confidence": round(confidence, 3)}


class HingBertClassifier:
    """
    Real classifier — loads a fine-tuned checkpoint.

    Usage (once you have a checkpoint from scripts/train.py):

        clf = HingBertClassifier(model_path="models/hingroberta-finetuned")
        clf.predict("yeh bakwas hai")

    Requires: pip install transformers torch (already in requirements.txt)
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        # must match the label order used in scripts/train.py's label encoding
        self.id2label = {0: "safe", 1: "abusive", 2: "severe"}

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0]
            pred_id = int(self.torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())

        return {"label": self.id2label[pred_id], "confidence": round(confidence, 3)}


def get_classifier():
    """
    Single place the consumer asks for a classifier instance.

    TODO (Week 3-4, per plan): once you've fine-tuned a model and have a checkpoint
    directory, swap the line below to:

        return HingBertClassifier(model_path="models/hingroberta-finetuned")
    """
    return RuleBasedClassifier()
