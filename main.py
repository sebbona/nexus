#!/usr/bin/env python3
"""
NexusTrader — Multi-Market AI Trading System
Crypto | Forex | XAUUSD | Stocks US
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

# ─── Logger Setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/nexus.log",
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_analyze(config: dict, args):
    """Lance l'analyse multi-agents complète."""
    from core.crew import NexusTradingCrew
    logger.info("Démarrage de l'analyse NexusTrader...")
    crew = NexusTradingCrew(config)
    result = crew.run_analysis()
    print("\n" + "═" * 70)
    print("RÉSULTAT DE L'ANALYSE")
    print("═" * 70)
    print(result)


def cmd_backtest(config: dict, args):
    """Lance le backtesting sur tous les marchés configurés."""
    from data.fetcher import MarketDataFetcher
    from core.backtester import NexusBacktester

    logger.info("Démarrage du backtesting...")
    config["data"]["lookback_days"] = 365  # plus de données pour backtest
    fetcher = MarketDataFetcher(config)
    backtester = NexusBacktester(config)

    logger.info("Récupération des données historiques...")
    data = fetcher.fetch_all()

    logger.info(f"Backtesting sur {len(data)} instruments...")
    results = backtester.run_all(data)
    backtester.print_summary(results)


def cmd_paper(config: dict, args):
    """Mode paper trading en continu avec schedule."""
    import schedule
    import time
    from core.crew import NexusTradingCrew
    from core.executor import TradeExecutor

    config["strategy"]["mode"] = "paper"
    crew = NexusTradingCrew(config)
    executor = TradeExecutor(config)

    def run_cycle():
        logger.info("Cycle paper trading...")
        result = crew.run_analysis()
        logger.info(f"Analyse: {result[:200]}...")

    interval_min = 60
    logger.info(f"Paper trading démarré — analyse toutes les {interval_min} minutes")
    run_cycle()
    schedule.every(interval_min).minutes.do(run_cycle)

    while True:
        schedule.run_pending()
        time.sleep(30)


def cmd_dashboard(config: dict, args):
    """Lance le dashboard Streamlit."""
    import subprocess
    port = config["dashboard"]["port"]
    host = config["dashboard"]["host"]
    logger.info(f"Dashboard sur http://{host}:{port}")
    subprocess.run(
        ["streamlit", "run", "dashboard/app.py", f"--server.port={port}", f"--server.address={host}"],
        check=True,
    )


def cmd_quick(config: dict, args):
    """Analyse rapide d'un instrument spécifique."""
    from core.crew import NexusTradingCrew
    crew = NexusTradingCrew(config)
    result = crew.run_quick_analysis(args.symbol, args.market)
    print(result)


def main():
    parser = argparse.ArgumentParser(
        description="NexusTrader — Système de Trading IA Multi-Marché",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py analyze                     # Analyse complète multi-agents
  python main.py backtest                    # Backtesting tous marchés
  python main.py paper                       # Paper trading en continu
  python main.py dashboard                   # Lance le dashboard
  python main.py quick --symbol BTC/USDT --market crypto
  python main.py quick --symbol GC=F --market commodities
  python main.py quick --symbol EURUSD=X --market forex
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Chemin vers config.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analyze", help="Analyse multi-agents complète")
    subparsers.add_parser("backtest", help="Backtesting sur données historiques")
    subparsers.add_parser("paper", help="Paper trading en continu")
    subparsers.add_parser("dashboard", help="Dashboard Streamlit")

    quick_parser = subparsers.add_parser("quick", help="Analyse rapide d'un instrument")
    quick_parser.add_argument("--symbol", required=True, help="Ex: BTC/USDT, EURUSD=X, GC=F, SPY")
    quick_parser.add_argument("--market", required=True,
                              choices=["crypto", "stocks", "forex", "commodities"],
                              help="Type de marché")

    args = parser.parse_args()
    config = load_config(args.config)

    commands = {
        "analyze": cmd_analyze,
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "dashboard": cmd_dashboard,
        "quick": cmd_quick,
    }
    commands[args.command](config, args)


if __name__ == "__main__":
    main()
