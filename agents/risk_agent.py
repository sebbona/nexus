"""Risk Manager Agent: gestion du risque, position sizing, drawdown protection."""

from __future__ import annotations

from dataclasses import dataclass
from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

from loguru import logger


@dataclass
class TradeSignal:
    symbol: str
    market: str
    action: str          # BUY | SELL | HOLD
    confidence: float    # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # % du capital
    approved: bool
    rejection_reason: str = ""


class RiskCheckInput(BaseModel):
    symbol: str = Field(description="Symbole du trade")
    action: str = Field(description="BUY | SELL | HOLD")
    confidence: float = Field(description="Confiance 0-1")
    entry_price: float = Field(description="Prix d'entrée")
    stop_loss: float = Field(description="Niveau stop-loss")
    take_profit: float = Field(description="Objectif take-profit")
    current_capital: float = Field(default=10000.0, description="Capital disponible en USD")
    open_positions: int = Field(default=0, description="Nombre de positions ouvertes")
    current_drawdown_pct: float = Field(default=0.0, description="Drawdown actuel en %")
    daily_loss_pct: float = Field(default=0.0, description="Perte journalière en %")


class RiskManagerTool(BaseTool):
    name: str = "risk_check"
    description: str = "Vérifie et valide un signal de trading selon les règles de gestion du risque"
    args_schema: type[BaseModel] = RiskCheckInput
    risk_config: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        symbol: str,
        action: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_capital: float = 10000.0,
        open_positions: int = 0,
        current_drawdown_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
    ) -> str:
        cfg = self.risk_config or {}
        max_pos_pct = cfg.get("max_position_pct", 0.05)
        max_drawdown = cfg.get("max_drawdown_pct", 0.15)
        max_positions = cfg.get("max_open_positions", 8)
        min_confidence = 0.60
        daily_limit = cfg.get("daily_loss_limit_pct", 0.05)

        rejections = []

        # Règle 1: Confiance minimale
        if confidence < min_confidence:
            rejections.append(f"Confiance {confidence:.0%} < minimum {min_confidence:.0%}")

        # Règle 2: HOLD ne tradé pas
        if action == "HOLD":
            return f"HOLD: Signal ignoré — pas de position ouverte pour {symbol}"

        # Règle 3: Drawdown maximum
        if current_drawdown_pct >= max_drawdown:
            rejections.append(f"Drawdown {current_drawdown_pct:.1%} >= limite {max_drawdown:.1%} — trading suspendu")

        # Règle 4: Limite journalière
        if daily_loss_pct >= daily_limit:
            rejections.append(f"Perte journalière {daily_loss_pct:.1%} >= limite {daily_limit:.1%}")

        # Règle 5: Positions maximum
        if open_positions >= max_positions:
            rejections.append(f"Positions ouvertes {open_positions}/{max_positions} — limite atteinte")

        # Règle 6: Ratio risque/récompense
        if action == "BUY":
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)

        rr_ratio = reward / risk if risk > 0 else 0
        if rr_ratio < 1.5:
            rejections.append(f"Ratio R/R {rr_ratio:.2f} < minimum 1.5")

        # Calcul du position sizing (Kelly Criterion simplifié)
        win_rate = confidence
        avg_win = reward / entry_price
        avg_loss = risk / entry_price
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
        kelly = max(0, min(kelly, max_pos_pct))  # cap au max configuré
        position_pct = kelly * 0.5  # demi-Kelly pour prudence
        position_usd = current_capital * position_pct

        if rejections:
            return (
                f"TRADE REJETÉ: {symbol}\n"
                f"{'─'*40}\n"
                f"Raisons:\n" + "\n".join(f"  ✗ {r}" for r in rejections)
            )

        stop_pct = abs(entry_price - stop_loss) / entry_price * 100
        tp_pct = abs(take_profit - entry_price) / entry_price * 100

        return (
            f"TRADE APPROUVÉ: {symbol}\n"
            f"{'─'*40}\n"
            f"Action: {action}\n"
            f"Taille position: {position_pct:.2%} du capital = {position_usd:.2f} USD\n"
            f"Kelly: {kelly:.2%} → Demi-Kelly: {position_pct:.2%}\n"
            f"Stop-Loss: {stop_loss:.5f} ({stop_pct:.2f}% risque)\n"
            f"Take-Profit: {take_profit:.5f} ({tp_pct:.2f}% gain)\n"
            f"Ratio R/R: 1:{rr_ratio:.2f}\n"
            f"Perte max possible: {position_usd * stop_pct / 100:.2f} USD ({position_usd * stop_pct / 100 / current_capital:.2%} capital)\n"
            f"Gain max potentiel: {position_usd * tp_pct / 100:.2f} USD\n"
            f"Positions après: {open_positions + 1}/{max_positions}"
        )


def create_risk_agent(risk_config: dict) -> tuple[Agent, Task]:
    tool = RiskManagerTool(risk_config=risk_config)

    agent = Agent(
        role="Chief Risk Manager",
        goal=(
            "Protéger le capital en validant chaque signal de trading selon des règles strictes: "
            "drawdown maximum, position sizing Kelly, ratio risque/récompense minimum, limites de positions"
        ),
        backstory=(
            "Tu es le risk manager d'un fonds d'investissement. Ta priorité absolue est la préservation du capital. "
            "Tu appliques le position sizing basé sur le critère de Kelly, tu surveilles le drawdown en temps réel, "
            "et tu peux couper toutes les positions si les limites de risque sont franchies. "
            "Tu n'approuves un trade que si le ratio risque/récompense est d'au moins 1:1.5 "
            "et si la confiance du stratégiste est suffisante."
        ),
        tools=[tool],
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Valide chaque signal de trading fourni par le stratégiste pour {symbols}. "
            "Pour chaque signal: vérifie la confiance, calcule le position sizing optimal (Kelly), "
            "valide le ratio R/R, et contrôle les limites de drawdown et de positions ouvertes. "
            "Approuve ou rejette avec justification détaillée."
        ),
        expected_output=(
            "Pour chaque instrument: décision d'approbation (APPROUVÉ/REJETÉ), "
            "taille de position exacte en USD et en %, "
            "niveaux définitifs de stop-loss et take-profit, "
            "ratio R/R calculé, et risque maximum en % du capital."
        ),
        agent=agent,
    )

    return agent, task
