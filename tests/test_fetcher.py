"""Tests for MarketDataFetcher — vérifie la récupération multi-marché."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


MOCK_CONFIG = {
    "exchanges": {"crypto": {"name": "binance", "api_key": "", "secret": "", "sandbox": True}},
    "markets": {
        "crypto": {"enabled": True, "symbols": ["BTC/USDT"], "timeframe": "1h"},
        "stocks": {"enabled": True, "symbols": ["SPY"], "timeframe": "1d"},
        "forex": {"enabled": True, "symbols": ["EURUSD=X"], "timeframe": "1h"},
        "commodities": {"enabled": True, "symbols": ["GC=F"], "timeframe": "1h"},
    },
    "data": {"lookback_days": 7, "cache_ttl_minutes": 5},
}


@pytest.mark.parametrize("symbol,market", [
    ("SPY", "stocks"),
    ("EURUSD=X", "forex"),
    ("GC=F", "commodities"),
])
def test_yfinance_fetch(symbol, market):
    from data.fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher(MOCK_CONFIG)
    df = fetcher.fetch_yfinance(symbol, market, timeframe="1d", lookback_days=7)
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame for {symbol}"
    assert not df.empty, f"Empty DataFrame for {symbol}"
    assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])


def test_cache():
    from data.fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher(MOCK_CONFIG)
    df1 = fetcher.fetch_yfinance("SPY", "stocks", timeframe="1d", lookback_days=7)
    df2 = fetcher.fetch_yfinance("SPY", "stocks", timeframe="1d", lookback_days=7)
    assert len(df1) == len(df2), "Cache should return same data"


def test_sentiment_aggregate():
    from models.sentiment import FinBERTSentiment
    model = FinBERTSentiment(device="cpu")
    result = model.aggregate_sentiment([])
    assert result.label == "neutral"
    assert result.compound == 0.0
