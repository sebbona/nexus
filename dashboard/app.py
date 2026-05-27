"""Streamlit dashboard — NexusTrader real-time monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="NexusTrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Dark Theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.signal-buy { color: #00ff88; font-weight: bold; font-size: 1.2em; }
.signal-sell { color: #ff4444; font-weight: bold; font-size: 1.2em; }
.signal-hold { color: #ffaa00; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_fetcher(config):
    from data.fetcher import MarketDataFetcher
    return MarketDataFetcher(config)


def load_trades() -> list[dict]:
    trades_file = Path("logs/trades.json")
    if not trades_file.exists():
        return []
    with open(trades_file) as f:
        return json.load(f)


def candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        )
    ])
    fig.update_layout(
        title=symbol,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        xaxis=dict(gridcolor="#21262d", rangeslider_visible=False),
        yaxis=dict(gridcolor="#21262d"),
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def portfolio_pnl_chart(trades: list[dict]) -> go.Figure:
    closed = [t for t in trades if t["status"] in ("CLOSED", "STOPPED")]
    if not closed:
        return go.Figure()

    df = pd.DataFrame(closed)
    df["closed_at"] = pd.to_datetime(df["closed_at"])
    df.sort_values("closed_at", inplace=True)
    df["cumulative_pnl"] = df["pnl_usd"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["closed_at"],
        y=df["cumulative_pnl"],
        mode="lines+markers",
        name="PnL cumulé",
        line=dict(color="#00ff88", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.1)",
    ))
    fig.update_layout(
        title="PnL Cumulé (USD)",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main():
    config = load_config()
    fetcher = get_fetcher(config)
    trades = load_trades()

    # ─── Header ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("NexusTrader")
        st.caption(f"Multi-Market AI Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        mode = config["strategy"]["mode"].upper()
        color = {"PAPER": "🟡", "LIVE": "🟢", "BACKTEST": "🔵"}.get(mode, "⚪")
        st.metric("Mode", f"{color} {mode}")
    with col3:
        open_pos = len([t for t in trades if t.get("status") == "OPEN"])
        st.metric("Positions ouvertes", open_pos)

    st.divider()

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Marchés actifs")
        for market, cfg in config["markets"].items():
            if cfg["enabled"]:
                label = {"crypto": "Crypto", "stocks": "Actions US", "forex": "Forex", "commodities": "Or/Matières"}.get(market, market)
                st.success(f"✓ {label}")

        st.divider()
        st.subheader("Risque")
        risk = config["risk"]
        st.metric("Max position", f"{risk['max_position_pct']:.0%} du capital")
        st.metric("Stop-loss", f"{risk['stop_loss_pct']:.0%}")
        st.metric("Take-profit", f"{risk['take_profit_pct']:.0%}")
        st.metric("Max drawdown", f"{risk['max_drawdown_pct']:.0%}")

        st.divider()
        refresh = st.slider("Rafraîchissement (s)", 10, 300, config["dashboard"]["refresh_seconds"])

    # ─── Portfolio Stats ──────────────────────────────────────────────────────
    closed_trades = [t for t in trades if t.get("status") in ("CLOSED", "STOPPED")]
    total_pnl = sum(t.get("pnl_usd", 0) or 0 for t in closed_trades)
    wins = [t for t in closed_trades if (t.get("pnl_usd") or 0) > 0]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PnL Total", f"${total_pnl:+,.2f}")
    c2.metric("Trades fermés", len(closed_trades))
    c3.metric("Win Rate", f"{win_rate:.1%}")
    c4.metric("Positions ouvertes", open_pos)
    c5.metric("Capital initial", f"${config['backtest']['initial_capital']:,}")

    # ─── PnL Chart ───────────────────────────────────────────────────────────
    if closed_trades:
        st.plotly_chart(portfolio_pnl_chart(trades), use_container_width=True)

    # ─── Market Charts ───────────────────────────────────────────────────────
    st.subheader("Marchés en temps réel")

    market_tabs = st.tabs(["Crypto", "Forex", "Or/Matières", "Actions US"])
    market_map = {
        0: ("crypto", config["markets"]["crypto"]["symbols"][:3]),
        1: ("forex", config["markets"]["forex"]["symbols"][:3]),
        2: ("commodities", config["markets"]["commodities"]["symbols"][:2]),
        3: ("stocks", config["markets"]["stocks"]["symbols"][:3]),
    }

    for tab_idx, tab in enumerate(market_tabs):
        with tab:
            market, symbols = market_map[tab_idx]
            if not config["markets"][market]["enabled"]:
                st.info(f"Marché {market} désactivé dans config.yaml")
                continue

            selected = st.selectbox(
                "Instrument",
                symbols,
                key=f"select_{market}",
            )
            with st.spinner(f"Chargement {selected}..."):
                try:
                    df = fetcher.fetch_symbol(market, selected)
                    if not df.empty:
                        last_close = df["close"].iloc[-1]
                        prev_close = df["close"].iloc[-2] if len(df) > 1 else last_close
                        change = (last_close - prev_close) / prev_close * 100
                        vol = df["volume"].tail(24).mean()

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Prix", f"{last_close:.5f}", f"{change:+.2f}%")
                        m2.metric("High 24h", f"{df['high'].tail(24).max():.5f}")
                        m3.metric("Vol moyen 24h", f"{vol:,.0f}")

                        st.plotly_chart(candlestick_chart(df.tail(200), selected), use_container_width=True)
                    else:
                        st.warning(f"Pas de données pour {selected}")
                except Exception as e:
                    st.error(f"Erreur: {e}")

    # ─── Open Positions Table ────────────────────────────────────────────────
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    if open_trades:
        st.subheader("Positions ouvertes")
        df_pos = pd.DataFrame(open_trades)[
            ["symbol", "market", "action", "entry_price", "stop_loss", "take_profit", "position_size_usd", "confidence", "opened_at"]
        ]
        df_pos.columns = ["Symbole", "Marché", "Action", "Entrée", "Stop-Loss", "Take-Profit", "Taille (USD)", "Confiance", "Ouvert le"]
        st.dataframe(df_pos, use_container_width=True)

    # ─── Trade History ───────────────────────────────────────────────────────
    if closed_trades:
        st.subheader("Historique des trades")
        df_hist = pd.DataFrame(closed_trades)
        display_cols = [c for c in ["symbol", "market", "action", "entry_price", "close_price", "pnl_usd", "pnl_pct", "status", "closed_at"] if c in df_hist.columns]
        df_hist = df_hist[display_cols].sort_values("closed_at", ascending=False)
        df_hist.columns = ["Symbole", "Marché", "Action", "Entrée", "Clôture", "PnL USD", "PnL %", "Statut", "Fermé le"][:len(display_cols)]
        st.dataframe(df_hist, use_container_width=True)

    # ─── Auto-refresh ────────────────────────────────────────────────────────
    st.caption(f"Prochain rafraîchissement dans {refresh}s")
    import time
    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()
