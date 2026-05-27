"""Multi-market data fetcher: Crypto (CCXT), Stocks, Forex, XAUUSD (yfinance)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "4h": "240m", "1d": "1d", "1w": "1wk",
}

MARKET_LABEL = {
    "crypto": "Crypto",
    "stocks": "Stocks US",
    "forex": "Forex",
    "commodities": "Or/Matières premières",
}


class MarketDataFetcher:
    def __init__(self, config: dict):
        self.config = config
        self._exchange: Optional[ccxt.Exchange] = None
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl = timedelta(minutes=config["data"].get("cache_ttl_minutes", 5))

    # ─── CCXT Exchange (Crypto) ───────────────────────────────────────────────

    def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            ex_cfg = self.config["exchanges"]["crypto"]
            exchange_class = getattr(ccxt, ex_cfg["name"])
            params = {"enableRateLimit": True}
            if ex_cfg.get("api_key"):
                params["apiKey"] = ex_cfg["api_key"]
                params["secret"] = ex_cfg["secret"]
            if ex_cfg.get("sandbox"):
                params["options"] = {"defaultType": "future"}
            self._exchange = exchange_class(params)
        return self._exchange

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def fetch_crypto(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV for a crypto pair via CCXT."""
        cache_key = f"crypto:{symbol}:{timeframe}"
        if cached := self._get_cached(cache_key):
            return cached

        exchange = self._get_exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.attrs["symbol"] = symbol
        df.attrs["market"] = "crypto"
        self._set_cache(cache_key, df)
        logger.debug(f"Crypto {symbol}: {len(df)} candles fetched")
        return df

    # ─── yfinance (Stocks, Forex, Or) ────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def fetch_yfinance(
        self,
        symbol: str,
        market: str,
        timeframe: str = "1h",
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """Fetch OHLCV via yfinance (stocks, forex EURUSD=X, gold GC=F)."""
        cache_key = f"{market}:{symbol}:{timeframe}"
        if cached := self._get_cached(cache_key):
            return cached

        yf_interval = TIMEFRAME_MAP.get(timeframe, "1h")
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, interval=yf_interval, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data for {symbol} ({market})")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.attrs["symbol"] = symbol
        df.attrs["market"] = market
        self._set_cache(cache_key, df)
        logger.debug(f"{MARKET_LABEL[market]} {symbol}: {len(df)} candles fetched")
        return df

    # ─── Fetch all configured markets ────────────────────────────────────────

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch data for all enabled markets from config."""
        results: dict[str, pd.DataFrame] = {}
        markets_cfg = self.config["markets"]
        lookback = self.config["data"].get("lookback_days", 30)

        if markets_cfg["crypto"]["enabled"]:
            tf = markets_cfg["crypto"]["timeframe"]
            for sym in markets_cfg["crypto"]["symbols"]:
                try:
                    results[f"crypto:{sym}"] = self.fetch_crypto(sym, tf)
                except Exception as e:
                    logger.error(f"Error fetching crypto {sym}: {e}")

        for market in ("stocks", "forex", "commodities"):
            if markets_cfg[market]["enabled"]:
                tf = markets_cfg[market]["timeframe"]
                for sym in markets_cfg[market]["symbols"]:
                    try:
                        results[f"{market}:{sym}"] = self.fetch_yfinance(
                            sym, market, tf, lookback
                        )
                    except Exception as e:
                        logger.error(f"Error fetching {market} {sym}: {e}")

        logger.info(f"Fetched {len(results)} instruments across all markets")
        return results

    def fetch_symbol(self, market: str, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch a single symbol by market type."""
        if market == "crypto":
            return self.fetch_crypto(symbol, **kwargs)
        return self.fetch_yfinance(symbol, market, **kwargs)

    # ─── Cache helpers ────────────────────────────────────────────────────────

    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._cache:
            df, ts = self._cache[key]
            if datetime.now() - ts < self._cache_ttl:
                return df
        return None

    def _set_cache(self, key: str, df: pd.DataFrame):
        self._cache[key] = (df, datetime.now())

    def clear_cache(self):
        self._cache.clear()
