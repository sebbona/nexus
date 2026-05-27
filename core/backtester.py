"""Backtesting engine via VectorBT — multi-marché, multi-stratégie."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from loguru import logger


@dataclass
class BacktestResult:
    symbol: str
    market: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float
    annualized_return_pct: float
    calmar_ratio: float
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float


class NexusBacktester:
    def __init__(self, config: dict):
        self.config = config
        self.bt_cfg = config["backtest"]
        self.initial_capital = self.bt_cfg["initial_capital"]
        self.commission = self.bt_cfg["commission"]

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: str,
        strategy_fn: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Run backtest on a DataFrame with OHLCV data.
        Uses the default NexusStrategy if no strategy_fn provided.
        """
        if df.empty or len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")

        entries, exits = self._generate_signals(df, strategy_fn)
        pf = self._run_portfolio(df, entries, exits)
        return self._extract_results(pf, symbol, market, df)

    def run_all(self, data: dict[str, pd.DataFrame]) -> list[BacktestResult]:
        """Backtest all instruments."""
        results = []
        for key, df in data.items():
            if df.empty:
                continue
            market, symbol = key.split(":", 1)
            try:
                result = self.run(df, symbol, market)
                results.append(result)
                logger.info(
                    f"Backtest {symbol}: Return={result.total_return_pct:+.1f}% | "
                    f"Sharpe={result.sharpe_ratio:.2f} | DD={result.max_drawdown_pct:.1f}%"
                )
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")
        return results

    def _generate_signals(self, df: pd.DataFrame, strategy_fn: Optional[Callable]) -> tuple:
        if strategy_fn:
            return strategy_fn(df)
        return self._default_strategy(df)

    def _default_strategy(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Default NexusStrategy:
        - Entry: RSI < 35 AND MACD crossover bullish AND close > EMA50
        - Exit: RSI > 65 OR MACD crossover bearish
        """
        close = df["close"]

        rsi = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)

        if macd_df is None or rsi is None:
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

        macd_line = macd_df["MACD_12_26_9"]
        macd_signal = macd_df["MACDs_12_26_9"]
        macd_cross_up = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
        macd_cross_down = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))

        trend_up = close > ema_50
        above_200 = close > ema_200 if ema_200 is not None else pd.Series(True, index=df.index)

        entries = (rsi < 35) & macd_cross_up & trend_up & above_200
        exits = (rsi > 65) | macd_cross_down

        return entries.fillna(False), exits.fillna(False)

    def _run_portfolio(self, df: pd.DataFrame, entries: pd.Series, exits: pd.Series) -> vbt.Portfolio:
        return vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission,
            freq="1H",
            sl_stop=self.config["risk"].get("stop_loss_pct", 0.02),
            tp_stop=self.config["risk"].get("take_profit_pct", 0.04),
        )

    def _extract_results(
        self, pf: vbt.Portfolio, symbol: str, market: str, df: pd.DataFrame
    ) -> BacktestResult:
        stats = pf.stats()
        total_return = float(pf.total_return()) * 100
        final_capital = self.initial_capital * (1 + float(pf.total_return()))

        sharpe = float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0.0
        max_dd = abs(float(pf.max_drawdown())) * 100
        win_rate = float(pf.trades.win_rate()) if len(pf.trades.records) > 0 else 0.0
        n_trades = int(pf.trades.count())

        profit_factor = 0.0
        try:
            wins = pf.trades.pnl[pf.trades.pnl > 0].sum()
            losses = abs(pf.trades.pnl[pf.trades.pnl < 0].sum())
            profit_factor = float(wins / losses) if losses > 0 else float("inf")
        except Exception:
            pass

        calmar = abs(total_return / max_dd) if max_dd > 0 else 0.0

        return BacktestResult(
            symbol=symbol,
            market=market,
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            win_rate=round(win_rate, 4),
            total_trades=n_trades,
            profit_factor=round(profit_factor, 3),
            annualized_return_pct=round(total_return / max(len(df) / 252, 1), 2),
            calmar_ratio=round(calmar, 3),
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            initial_capital=self.initial_capital,
            final_capital=round(final_capital, 2),
        )

    def print_summary(self, results: list[BacktestResult]):
        print(f"\n{'═'*70}")
        print(f"{'NEXUS BACKTEST SUMMARY':^70}")
        print(f"{'═'*70}")
        print(f"{'Symbole':<15} {'Marché':<12} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinRate':>8} {'Trades':>7}")
        print(f"{'─'*70}")
        for r in sorted(results, key=lambda x: x.total_return_pct, reverse=True):
            print(
                f"{r.symbol:<15} {r.market:<12} "
                f"{r.total_return_pct:>+7.1f}% {r.sharpe_ratio:>7.2f} "
                f"{r.max_drawdown_pct:>6.1f}% {r.win_rate:>7.1%} {r.total_trades:>7}"
            )
        print(f"{'═'*70}\n")
