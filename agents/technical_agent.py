"""Technical Agent: RSI, MACD, Bollinger, EMA, ATR, Volume Profile."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

from data.fetcher import MarketDataFetcher


class TechnicalInput(BaseModel):
    market: str = Field(description="Type de marché: crypto, stocks, forex, commodities")
    symbol: str = Field(description="Symbole à analyser")


class TechnicalAnalysisTool(BaseTool):
    name: str = "technical_analysis"
    description: str = "Calcule les indicateurs techniques: RSI, MACD, Bollinger Bands, EMA, ATR pour détecter des signaux"
    args_schema: type[BaseModel] = TechnicalInput
    fetcher: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, market: str, symbol: str) -> str:
        df = self.fetcher.fetch_symbol(market, symbol)
        if df.empty or len(df) < 50:
            return f"Données insuffisantes pour l'analyse technique de {symbol}"

        signals = self._compute_indicators(df)
        return self._format_report(symbol, df, signals)

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rsi = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        bb = ta.bbands(close, length=20, std=2)
        ema_20 = ta.ema(close, length=20)
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        atr = ta.atr(high, low, close, length=14)

        last_rsi = float(rsi.iloc[-1]) if rsi is not None else 50
        last_macd = float(macd["MACD_12_26_9"].iloc[-1]) if macd is not None else 0
        last_macd_signal = float(macd["MACDs_12_26_9"].iloc[-1]) if macd is not None else 0
        last_macd_hist = float(macd["MACDh_12_26_9"].iloc[-1]) if macd is not None else 0

        bb_upper = float(bb["BBU_20_2.0"].iloc[-1]) if bb is not None else None
        bb_lower = float(bb["BBL_20_2.0"].iloc[-1]) if bb is not None else None
        bb_mid = float(bb["BBM_20_2.0"].iloc[-1]) if bb is not None else None

        last_ema20 = float(ema_20.iloc[-1]) if ema_20 is not None else None
        last_ema50 = float(ema_50.iloc[-1]) if ema_50 is not None else None
        last_ema200 = float(ema_200.iloc[-1]) if ema_200 is not None else None
        last_atr = float(atr.iloc[-1]) if atr is not None else None
        last_close = float(close.iloc[-1])

        # Signal composite
        bullish_signals = 0
        bearish_signals = 0

        if last_rsi < 30:
            bullish_signals += 2  # survente forte
        elif last_rsi < 40:
            bullish_signals += 1
        elif last_rsi > 70:
            bearish_signals += 2  # surachat fort
        elif last_rsi > 60:
            bearish_signals += 1

        if last_macd > last_macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if last_macd_hist > 0 and last_macd_hist > float(macd["MACDh_12_26_9"].iloc[-2] or 0):
            bullish_signals += 1
        elif last_macd_hist < 0:
            bearish_signals += 1

        if last_ema20 and last_ema50:
            if last_ema20 > last_ema50:
                bullish_signals += 1
            else:
                bearish_signals += 1

        if last_ema200 and last_close > last_ema200:
            bullish_signals += 1
        elif last_ema200 and last_close < last_ema200:
            bearish_signals += 1

        if bb_lower and last_close < bb_lower:
            bullish_signals += 1
        elif bb_upper and last_close > bb_upper:
            bearish_signals += 1

        total = bullish_signals + bearish_signals
        bull_score = bullish_signals / total if total > 0 else 0.5

        return {
            "rsi": last_rsi,
            "macd": last_macd,
            "macd_signal": last_macd_signal,
            "macd_hist": last_macd_hist,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_mid": bb_mid,
            "ema_20": last_ema20,
            "ema_50": last_ema50,
            "ema_200": last_ema200,
            "atr": last_atr,
            "close": last_close,
            "bull_score": bull_score,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
        }

    def _format_report(self, symbol: str, df: pd.DataFrame, s: dict) -> str:
        trend = "HAUSSIÈRE" if s["bull_score"] > 0.6 else "BAISSIÈRE" if s["bull_score"] < 0.4 else "NEUTRE/LATÉRALE"
        rsi_label = "SURVENDU" if s["rsi"] < 30 else "SURACHETÉ" if s["rsi"] > 70 else "NEUTRE"
        macd_label = "HAUSSIER" if s["macd"] > s["macd_signal"] else "BAISSIER"

        atr_pct = (s["atr"] / s["close"] * 100) if s["atr"] and s["close"] else 0

        return (
            f"Analyse Technique: {symbol}\n"
            f"{'─'*50}\n"
            f"Prix: {s['close']:.5f}\n"
            f"Tendance: {trend} ({s['bullish_signals']} haussier / {s['bearish_signals']} baissier)\n\n"
            f"Indicateurs:\n"
            f"  RSI(14): {s['rsi']:.1f} → {rsi_label}\n"
            f"  MACD: {s['macd']:.5f} | Signal: {s['macd_signal']:.5f} → {macd_label}\n"
            f"  Histogramme MACD: {s['macd_hist']:+.5f}\n"
            f"  EMA 20/50/200: {s['ema_20']:.4f} / {s['ema_50']:.4f} / {s.get('ema_200', 'N/A')}\n"
            f"  Bollinger: Upper={s['bb_upper']:.4f} | Mid={s['bb_mid']:.4f} | Lower={s['bb_lower']:.4f}\n"
            f"  ATR(14): {s['atr']:.5f} ({atr_pct:.2f}% du prix) → Volatilité\n\n"
            f"Score technique: {s['bull_score']:.1%} haussier\n"
            f"Bougies analysées: {len(df)}"
        )


def create_technical_agent(fetcher: MarketDataFetcher) -> tuple[Agent, Task]:
    tool = TechnicalAnalysisTool(fetcher=fetcher)

    agent = Agent(
        role="Technical Analysis Expert",
        goal="Identifier les signaux techniques d'achat et de vente sur tous les marchés (crypto, forex, stocks, or) via RSI, MACD, Bollinger Bands, EMA",
        backstory=(
            "Tu es un analyste technique chevronné avec 15 ans d'expérience sur les marchés financiers. "
            "Tu maîtrises l'analyse technique classique et moderne: indicateurs de momentum (RSI), "
            "de tendance (MACD, EMA), de volatilité (ATR, Bollinger). "
            "Tu sais identifier les zones de support/résistance et les patterns de retournement."
        ),
        tools=[tool],
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Effectue une analyse technique complète pour chaque instrument: {symbols}. "
            "Calcule RSI, MACD, Bollinger Bands, EMA 20/50/200 et ATR. "
            "Identifie les signaux forts (divergences RSI, croisements MACD, sortie de Bollinger). "
            "Détermine si chaque instrument est en zone de surachat/survente."
        ),
        expected_output=(
            "Pour chaque instrument: les valeurs des indicateurs clés, les signaux détectés "
            "(fort/faible, haussier/baissier), les niveaux de support/résistance techniques, "
            "et un score de confiance technique (0-100%)."
        ),
        agent=agent,
    )

    return agent, task
