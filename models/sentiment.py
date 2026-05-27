"""FinBERT sentiment analysis wrapper for financial news."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class SentimentResult:
    label: str          # positive | negative | neutral
    score: float        # confidence 0-1
    compound: float     # -1 (very negative) to +1 (very positive)


class FinBERTSentiment:
    """
    Wraps ProsusAI/finbert for financial sentiment analysis.
    Lazy-loads the model on first use to avoid startup delay.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            logger.info(f"Loading FinBERT model ({self.MODEL_NAME})...")
            self._pipeline = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                device=-1 if self.device == "cpu" else 0,
                top_k=None,
            )
            logger.success("FinBERT loaded successfully")
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers torch")
            raise

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        self._load()
        text = text[:512]  # BERT token limit
        results = self._pipeline(text)[0]
        scores = {r["label"].lower(): r["score"] for r in results}
        top = max(results, key=lambda x: x["score"])
        compound = scores.get("positive", 0) - scores.get("negative", 0)
        return SentimentResult(
            label=top["label"].lower(),
            score=top["score"],
            compound=round(compound, 4),
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze a batch of texts efficiently."""
        self._load()
        truncated = [t[:512] for t in texts]
        all_results = self._pipeline(truncated, batch_size=self.batch_size)
        output = []
        for results in all_results:
            scores = {r["label"].lower(): r["score"] for r in results}
            top = max(results, key=lambda x: x["score"])
            compound = scores.get("positive", 0) - scores.get("negative", 0)
            output.append(SentimentResult(
                label=top["label"].lower(),
                score=top["score"],
                compound=round(compound, 4),
            ))
        return output

    def aggregate_sentiment(self, texts: list[str]) -> SentimentResult:
        """Aggregate sentiment score across multiple texts (e.g. all news for one symbol)."""
        if not texts:
            return SentimentResult(label="neutral", score=1.0, compound=0.0)
        results = self.analyze_batch(texts)
        avg_compound = sum(r.compound for r in results) / len(results)
        if avg_compound > 0.1:
            label = "positive"
        elif avg_compound < -0.1:
            label = "negative"
        else:
            label = "neutral"
        avg_score = sum(r.score for r in results) / len(results)
        return SentimentResult(label=label, score=avg_score, compound=round(avg_compound, 4))
