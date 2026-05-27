"""News fetcher: RSS feeds + CryptoPanic for sentiment analysis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import re

import aiohttp
import feedparser
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


SYMBOL_KEYWORDS: dict[str, list[str]] = {
    # Crypto
    "BTC/USDT": ["bitcoin", "btc", "crypto"],
    "ETH/USDT": ["ethereum", "eth", "ether"],
    "SOL/USDT": ["solana", "sol"],
    # Forex
    "EURUSD=X": ["euro", "eur", "usd", "dollar", "ecb", "fed"],
    "GBPUSD=X": ["pound", "gbp", "sterling", "boe", "bank of england"],
    "USDJPY=X": ["yen", "jpy", "boj", "bank of japan"],
    # Gold
    "GC=F": ["gold", "xauusd", "precious metal", "inflation", "fed"],
    # Stocks
    "SPY": ["s&p 500", "sp500", "market", "nasdaq", "dow"],
    "NVDA": ["nvidia", "nvda", "gpu", "ai chip"],
    "AAPL": ["apple", "aapl", "iphone", "tim cook"],
}

CRYPTO_RSS = "https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies={currencies}&kind=news"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
INVESTING_RSS = "https://www.investing.com/rss/news.rss"


@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    published: datetime
    source: str
    symbols: list[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None


class NewsFetcher:
    def __init__(self, config: dict):
        self.config = config
        self._cache: dict[str, tuple[list[NewsItem], datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)

    def fetch_all_news(self, symbols: Optional[list[str]] = None) -> list[NewsItem]:
        """Fetch news from all sources and filter by symbols."""
        all_news: list[NewsItem] = []

        all_news.extend(self._fetch_yahoo_rss(symbols))
        all_news.extend(self._fetch_cryptopanic())
        all_news.extend(self._fetch_investing_rss())

        # deduplicate by title
        seen = set()
        unique = []
        for item in all_news:
            if item.title not in seen:
                seen.add(item.title)
                if symbols:
                    item.symbols = self._match_symbols(item, symbols)
                unique.append(item)

        unique.sort(key=lambda x: x.published, reverse=True)
        logger.info(f"Fetched {len(unique)} unique news items")
        return unique

    def _fetch_yahoo_rss(self, symbols: Optional[list[str]] = None) -> list[NewsItem]:
        items = []
        targets = symbols or ["BTC-USD", "GC=F", "SPY", "EURUSD=X"]
        for sym in targets[:5]:  # limit requests
            try:
                url = YAHOO_RSS.format(symbol=sym.replace("/", "-"))
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    published = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now()
                    items.append(NewsItem(
                        title=entry.get("title", ""),
                        summary=entry.get("summary", "")[:500],
                        url=entry.get("link", ""),
                        published=published,
                        source="Yahoo Finance",
                    ))
            except Exception as e:
                logger.warning(f"Yahoo RSS error for {sym}: {e}")
        return items

    def _fetch_cryptopanic(self) -> list[NewsItem]:
        items = []
        try:
            url = CRYPTO_RSS.format(currencies="BTC,ETH,SOL")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get("results", [])[:20]:
                    items.append(NewsItem(
                        title=post.get("title", ""),
                        summary=post.get("title", ""),
                        url=post.get("url", ""),
                        published=datetime.fromisoformat(
                            post.get("published_at", datetime.now().isoformat()).replace("Z", "+00:00")
                        ).replace(tzinfo=None),
                        source="CryptoPanic",
                        symbols=["BTC/USDT", "ETH/USDT"],
                    ))
        except Exception as e:
            logger.warning(f"CryptoPanic error: {e}")
        return items

    def _fetch_investing_rss(self) -> list[NewsItem]:
        items = []
        try:
            feed = feedparser.parse(INVESTING_RSS)
            for entry in feed.entries[:15]:
                published = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now()
                items.append(NewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", "")[:500],
                    url=entry.get("link", ""),
                    published=published,
                    source="Investing.com",
                ))
        except Exception as e:
            logger.warning(f"Investing.com RSS error: {e}")
        return items

    def _match_symbols(self, item: NewsItem, symbols: list[str]) -> list[str]:
        """Match news to trading symbols based on keywords."""
        text = (item.title + " " + item.summary).lower()
        matched = []
        for sym in symbols:
            keywords = SYMBOL_KEYWORDS.get(sym, [sym.lower().replace("/usdt", "").replace("=x", "")])
            if any(kw in text for kw in keywords):
                matched.append(sym)
        return matched

    def get_symbol_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Get news specific to one symbol."""
        all_news = self.fetch_all_news([symbol])
        relevant = [n for n in all_news if not n.symbols or symbol in n.symbols]
        return relevant[:limit]
