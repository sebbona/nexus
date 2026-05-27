"""Sentiment Agent: analyse FinBERT sur les news financières."""

from __future__ import annotations

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

from data.news import NewsFetcher
from models.sentiment import FinBERTSentiment


class SentimentInput(BaseModel):
    symbol: str = Field(description="Symbole à analyser ex: BTC/USDT, EURUSD=X, GC=F")
    limit: int = Field(default=15, description="Nombre de news à analyser")


class SentimentAnalysisTool(BaseTool):
    name: str = "analyze_sentiment"
    description: str = "Analyse le sentiment des news financières récentes pour un instrument via FinBERT"
    args_schema: type[BaseModel] = SentimentInput
    news_fetcher: Any = None
    sentiment_model: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, symbol: str, limit: int = 15) -> str:
        news = self.news_fetcher.get_symbol_news(symbol, limit=limit)
        if not news:
            return f"Aucune news récente pour {symbol}. Sentiment: NEUTRE (0.0)"

        texts = [f"{n.title}. {n.summary}" for n in news]
        aggregate = self.sentiment_model.aggregate_sentiment(texts)
        individual = self.sentiment_model.analyze_batch(texts[:5])

        headlines = "\n".join(f"  • [{r.label.upper()}] {news[i].title}" for i, r in enumerate(individual))
        bar = self._sentiment_bar(aggregate.compound)

        return (
            f"Sentiment pour {symbol} ({len(texts)} news analysées)\n"
            f"Score agrégé: {aggregate.compound:+.3f} {bar}\n"
            f"Label: {aggregate.label.upper()} (confiance: {aggregate.score:.1%})\n\n"
            f"Derniers titres analysés:\n{headlines}"
        )

    def _sentiment_bar(self, compound: float) -> str:
        filled = int((compound + 1) / 2 * 10)
        return "[" + "█" * filled + "░" * (10 - filled) + f"] {compound:+.2f}"


def create_sentiment_agent(news_fetcher: NewsFetcher, sentiment_model: FinBERTSentiment) -> tuple[Agent, Task]:
    tool = SentimentAnalysisTool(news_fetcher=news_fetcher, sentiment_model=sentiment_model)

    agent = Agent(
        role="Financial Sentiment Analyst",
        goal="Analyser le sentiment du marché via les news financières en utilisant FinBERT pour identifier la pression acheteuse ou vendeuse",
        backstory=(
            "Tu es un analyste spécialisé en traitement du langage naturel financier. "
            "Tu utilises FinBERT, un modèle entraîné sur des millions de textes financiers, "
            "pour détecter le sentiment positif/négatif/neutre dans les news et rapports de marché. "
            "Tu sais que le sentiment peut anticiper les mouvements de prix de quelques heures."
        ),
        tools=[tool],
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Analyse le sentiment des news récentes pour chaque instrument: {symbols}. "
            "Pour chaque instrument, lis les dernières news et calcule un score de sentiment agrégé. "
            "Identifie les catalyseurs positifs ou négatifs majeurs qui pourraient impacter les prix."
        ),
        expected_output=(
            "Pour chaque instrument: score de sentiment (-1 à +1), label (positif/négatif/neutre), "
            "les 3 catalyseurs principaux identifiés, et une recommandation de biais directionnel."
        ),
        agent=agent,
    )

    return agent, task
