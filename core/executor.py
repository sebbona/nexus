"""Signal executor: paper trading ou live via CCXT / yfinance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class Trade:
    id: str
    symbol: str
    market: str
    action: str           # BUY | SELL
    entry_price: float
    position_size_usd: float
    stop_loss: float
    take_profit: float
    confidence: float
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())
    closed_at: Optional[str] = None
    close_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "OPEN"  # OPEN | CLOSED | STOPPED


class TradeExecutor:
    """
    Paper trading executor with persistent trade log.
    Supporte live trading via CCXT si mode = 'live'.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mode = config["strategy"]["mode"]  # paper | live | backtest
        self.trades_file = Path("logs/trades.json")
        self.trades_file.parent.mkdir(exist_ok=True)
        self.open_trades: list[Trade] = self._load_trades()

    def execute_signal(
        self,
        symbol: str,
        market: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size_usd: float,
        confidence: float,
    ) -> Optional[Trade]:
        if action == "HOLD":
            logger.info(f"HOLD {symbol} — aucune action")
            return None

        if self.mode == "paper":
            return self._paper_execute(
                symbol, market, action, entry_price,
                stop_loss, take_profit, position_size_usd, confidence
            )
        elif self.mode == "live":
            return self._live_execute(
                symbol, market, action, entry_price,
                stop_loss, take_profit, position_size_usd, confidence
            )
        else:
            logger.warning(f"Mode '{self.mode}' — exécution désactivée en mode backtest")
            return None

    def _paper_execute(self, symbol, market, action, entry_price, stop_loss, take_profit, size_usd, confidence) -> Trade:
        trade = Trade(
            id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            market=market,
            action=action,
            entry_price=entry_price,
            position_size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
        )
        self.open_trades.append(trade)
        self._save_trades()
        logger.success(
            f"[PAPER] {action} {symbol} @ {entry_price:.5f} | "
            f"Size: ${size_usd:.2f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}"
        )
        return trade

    def _live_execute(self, symbol, market, action, entry_price, stop_loss, take_profit, size_usd, confidence) -> Optional[Trade]:
        if market != "crypto":
            logger.warning(f"Live trading uniquement disponible pour crypto. {market} ignoré.")
            return self._paper_execute(symbol, market, action, entry_price, stop_loss, take_profit, size_usd, confidence)
        try:
            import ccxt
            ex_cfg = self.config["exchanges"]["crypto"]
            exchange = getattr(ccxt, ex_cfg["name"])({
                "apiKey": ex_cfg["api_key"],
                "secret": ex_cfg["secret"],
                "enableRateLimit": True,
            })
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
            amount = size_usd / current_price
            side = "buy" if action == "BUY" else "sell"
            order = exchange.create_market_order(symbol, side, amount)
            logger.success(f"[LIVE] Order placed: {order}")
            trade = Trade(
                id=str(order["id"]),
                symbol=symbol,
                market=market,
                action=action,
                entry_price=current_price,
                position_size_usd=size_usd,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
            )
            self.open_trades.append(trade)
            self._save_trades()
            return trade
        except Exception as e:
            logger.error(f"Live execution failed: {e}. Fallback to paper.")
            return self._paper_execute(symbol, market, action, entry_price, stop_loss, take_profit, size_usd, confidence)

    def update_positions(self, current_prices: dict[str, float]):
        """Check stop-loss and take-profit for all open positions."""
        closed = []
        for trade in self.open_trades:
            if trade.status != "OPEN":
                continue
            price = current_prices.get(f"{trade.market}:{trade.symbol}")
            if price is None:
                continue

            if trade.action == "BUY":
                if price <= trade.stop_loss:
                    self._close_trade(trade, price, "STOPPED")
                    closed.append(trade)
                elif price >= trade.take_profit:
                    self._close_trade(trade, price, "CLOSED")
                    closed.append(trade)
            else:
                if price >= trade.stop_loss:
                    self._close_trade(trade, price, "STOPPED")
                    closed.append(trade)
                elif price <= trade.take_profit:
                    self._close_trade(trade, price, "CLOSED")
                    closed.append(trade)

        self.open_trades = [t for t in self.open_trades if t.status == "OPEN"]
        if closed:
            self._save_trades()

    def _close_trade(self, trade: Trade, close_price: float, status: str):
        trade.closed_at = datetime.now().isoformat()
        trade.close_price = close_price
        trade.status = status
        if trade.action == "BUY":
            trade.pnl_pct = (close_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl_pct = (trade.entry_price - close_price) / trade.entry_price
        trade.pnl_usd = trade.position_size_usd * trade.pnl_pct

        emoji = "✅" if trade.pnl_usd > 0 else "❌"
        logger.info(
            f"{emoji} [{status}] {trade.symbol}: "
            f"PnL = {trade.pnl_usd:+.2f} USD ({trade.pnl_pct:+.2%})"
        )

    def get_portfolio_stats(self) -> dict:
        all_trades = self._load_all_trades()
        closed = [t for t in all_trades if t.status in ("CLOSED", "STOPPED")]
        if not closed:
            return {"total_trades": 0, "total_pnl": 0, "win_rate": 0}

        total_pnl = sum(t.pnl_usd or 0 for t in closed)
        wins = [t for t in closed if (t.pnl_usd or 0) > 0]
        return {
            "total_trades": len(closed),
            "open_trades": len(self.open_trades),
            "total_pnl_usd": round(total_pnl, 2),
            "win_rate": round(len(wins) / len(closed), 4) if closed else 0,
            "avg_win_usd": round(sum(t.pnl_usd or 0 for t in wins) / len(wins), 2) if wins else 0,
        }

    def _save_trades(self):
        all_trades = self._load_all_trades()
        ids = {t.id for t in all_trades}
        for t in self.open_trades:
            if t.id not in ids:
                all_trades.append(t)
        with open(self.trades_file, "w") as f:
            json.dump([asdict(t) for t in all_trades], f, indent=2, default=str)

    def _load_trades(self) -> list[Trade]:
        return [t for t in self._load_all_trades() if t.status == "OPEN"]

    def _load_all_trades(self) -> list[Trade]:
        if not self.trades_file.exists():
            return []
        try:
            with open(self.trades_file) as f:
                data = json.load(f)
            return [Trade(**d) for d in data]
        except Exception:
            return []
