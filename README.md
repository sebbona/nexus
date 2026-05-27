# NexusTrader

Système de trading algorithmique multi-marché piloté par 5 agents IA collaboratifs.

## Marchés supportés

| Marché | Exemples | Source |
|--------|----------|--------|
| **Crypto** | BTC/USDT, ETH/USDT, SOL/USDT | CCXT (Binance, Bybit...) |
| **Forex** | EUR/USD, GBP/USD, USD/JPY | yfinance |
| **Or/XAUUSD** | GC=F (Gold Futures) | yfinance |
| **Actions US** | SPY, QQQ, NVDA, AAPL | yfinance |

## Architecture — 5 Agents IA

```
[Data Agent] ──────────────┐
                           ▼
[Technical Agent] ────→ [Strategy Agent (DeepSeek LLM)]
                           │
[Sentiment Agent] ─────────┘
(FinBERT)                  │
                           ▼
                  [Risk Manager Agent]
                    (Kelly Criterion)
                           │
                           ▼
                  BUY / SELL / HOLD
                    + Position Size
```

| Agent | Rôle | Modèle |
|-------|------|--------|
| **Data Agent** | Collecte OHLCV temps réel | CCXT + yfinance |
| **Sentiment Agent** | Analyse news financières | FinBERT (HuggingFace) |
| **Technical Agent** | RSI, MACD, Bollinger, EMA | pandas-ta |
| **Strategy Agent** | Décision BUY/SELL/HOLD | DeepSeek-V3 (LLM) |
| **Risk Manager** | Position sizing, drawdown | Kelly Criterion |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Édite `config.yaml` et renseigne tes clés API :

```yaml
api_keys:
  deepseek: "sk-..."        # https://platform.deepseek.com
exchanges:
  crypto:
    name: "binance"
    api_key: ""             # optionnel pour paper trading
    sandbox: true
```

> **DeepSeek** : gratuit jusqu'à 10M tokens/mois.  
> Tous les marchés fonctionnent sans clé API (yfinance est gratuit).

## Utilisation

```bash
# Analyse complète multi-agents (tous marchés)
python main.py analyze

# Backtesting sur données historiques 2023-2024
python main.py backtest

# Paper trading en continu (toutes les 60min)
python main.py paper

# Dashboard temps réel
python main.py dashboard
# → http://localhost:8501

# Analyse rapide d'un instrument
python main.py quick --symbol GC=F --market commodities      # Or
python main.py quick --symbol EURUSD=X --market forex        # EUR/USD
python main.py quick --symbol BTC/USDT --market crypto       # Bitcoin
python main.py quick --symbol SPY --market stocks            # S&P 500
```

## Stratégie par défaut

**Entrée :** RSI(14) < 35 + croisement MACD haussier + close > EMA50  
**Sortie :** RSI > 65 OU croisement MACD baissier  
**Stop-loss :** 2% | **Take-profit :** 4% | **Ratio R/R :** 1:2  
**Position sizing :** Demi-critère de Kelly (capital-safe)

## Modes

| Mode | Description |
|------|-------------|
| `paper` | Simulation sans argent réel (défaut) |
| `live` | Trading réel via CCXT (crypto uniquement) |
| `backtest` | Test sur données historiques |

## Stack technique

- **LLM** : DeepSeek-V3 (gratuit) ou Ollama local (Llama 3.3)
- **Sentiment** : ProsusAI/FinBERT (HuggingFace)
- **Orchestration** : CrewAI
- **Backtesting** : VectorBT
- **Data** : CCXT + yfinance
- **Dashboard** : Streamlit + Plotly

## Tests

```bash
pytest tests/ -v
```
