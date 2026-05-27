"""Data Agent: collecte et prépare les données OHLCV pour tous les marchés."""

from __future__ import annotations

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

from data.fetcher import MarketDataFetcher


class FetchMarketDataInput(BaseModel):
    market: str = Field(description="Type de marché: crypto, stocks, forex, commodities")
    symbol: str = Field(description="Symbole ex: BTC/USDT, EURUSD=X, GC=F, SPY")
    timeframe: str = Field(default="1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d")


class FetchMarketDataTool(BaseTool):
    name: str = "fetch_market_data"
    description: str = "Récupère les données OHLCV pour un instrument financier (crypto, forex, actions, or)"
    args_schema: type[BaseModel] = FetchMarketDataInput
    fetcher: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, market: str, symbol: str, timeframe: str = "1h") -> str:
        df = self.fetcher.fetch_symbol(market, symbol, timeframe=timeframe)
        if df.empty:
            return f"Aucune donnée disponible pour {symbol}"
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        change_pct = ((last["close"] - prev["close"]) / prev["close"]) * 100
        return (
            f"Marché: {market.upper()} | Symbole: {symbol}\n"
            f"Dernière clôture: {last['close']:.5f}\n"
            f"Variation: {change_pct:+.2f}%\n"
            f"High 24h: {df['high'].tail(24).max():.5f}\n"
            f"Low 24h: {df['low'].tail(24).min():.5f}\n"
            f"Volume moyen: {df['volume'].tail(24).mean():.2f}\n"
            f"Bougies disponibles: {len(df)}"
        )


def create_data_agent(fetcher: MarketDataFetcher) -> tuple[Agent, Task]:
    tool = FetchMarketDataTool(fetcher=fetcher)

    agent = Agent(
        role="Market Data Specialist",
        goal="Collecter et synthétiser les données de marché pour tous les instruments: crypto, forex, actions, or (XAUUSD)",
        backstory=(
            "Tu es un spécialiste des données de marché financier avec accès à tous les marchés mondiaux. "
            "Tu collectes les données OHLCV en temps réel et produis un résumé clair de l'état actuel "
            "des marchés pour aider l'équipe de trading à prendre des décisions éclairées."
        ),
        tools=[tool],
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Récupère et analyse les données de marché pour les instruments configurés. "
            "Pour chaque marché (crypto, forex, actions, commodités), fournis: "
            "le prix actuel, la variation, les niveaux clés (high/low), et la tendance générale. "
            "Instruments à analyser: {symbols}"
        ),
        expected_output=(
            "Un rapport structuré avec pour chaque instrument: prix, variation %, tendance (haussière/baissière/latérale), "
            "niveaux de support/résistance basés sur high/low récents."
        ),
        agent=agent,
    )

    return agent, task
