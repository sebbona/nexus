"""Strategy Agent: raisonnement LLM (DeepSeek) pour décision finale BUY/SELL/HOLD."""

from __future__ import annotations

import json
from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Any

from loguru import logger


class LLMReasoningInput(BaseModel):
    symbol: str = Field(description="Symbole à analyser")
    market: str = Field(description="Type de marché")
    technical_report: str = Field(description="Rapport de l'agent technique")
    sentiment_report: str = Field(description="Rapport de l'agent sentiment")
    data_report: str = Field(description="Rapport de l'agent données")


SYSTEM_PROMPT = """Tu es un trader quant senior avec 20 ans d'expérience sur les marchés financiers.
Tu analyses des données techniques et fondamentales pour prendre des décisions de trading précises.
Tu dois être rationnel, quantitatif et concis. Tu gères le risque avec rigueur.
Tes décisions sont basées sur la convergence des signaux (technique + sentiment + macro).
Tu réponds TOUJOURS en JSON structuré."""

DECISION_PROMPT = """Analyse complète pour {symbol} ({market}):

=== DONNÉES DE MARCHÉ ===
{data_report}

=== ANALYSE TECHNIQUE ===
{technical_report}

=== SENTIMENT NEWS ===
{sentiment_report}

=== CONTEXTE MACRO ===
- Considère la corrélation inter-marchés (risk-on/risk-off)
- Pour XAUUSD/GC=F: inflation, taux Fed, géopolitique
- Pour Forex: différentiels de taux, balance commerciale
- Pour Crypto: dominance BTC, flux institutionnels
- Pour Actions: earnings, macro US

Prends une décision de trading et réponds en JSON:
{{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0 à 1.0,
    "reasoning": "Explication concise en 2-3 phrases",
    "key_signals": ["signal1", "signal2", "signal3"],
    "risks": ["risque1", "risque2"],
    "entry_zone": "niveau ou zone d'entrée suggérée",
    "stop_loss_hint": "niveau stop suggéré",
    "take_profit_hint": "objectif de prix",
    "timeframe": "court terme (1-4h) | moyen terme (1-3j) | long terme (1-2sem)"
}}"""


class LLMStrategyTool(BaseTool):
    name: str = "llm_strategy_decision"
    description: str = "Utilise DeepSeek LLM pour analyser tous les signaux et prendre une décision BUY/SELL/HOLD avec raisonnement"
    args_schema: type[BaseModel] = LLMReasoningInput
    llm_config: Any = None
    client: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context):
        cfg = self.llm_config or {}
        provider = cfg.get("provider", "deepseek")
        if provider == "deepseek":
            base_url = "https://api.deepseek.com"
            api_key = cfg.get("api_key", "sk-placeholder")
        elif provider == "ollama":
            base_url = cfg.get("ollama_base_url", "http://localhost:11434/v1")
            api_key = "ollama"
        else:
            base_url = "https://api.openai.com/v1"
            api_key = cfg.get("api_key", "")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _run(
        self,
        symbol: str,
        market: str,
        technical_report: str,
        sentiment_report: str,
        data_report: str,
    ) -> str:
        prompt = DECISION_PROMPT.format(
            symbol=symbol,
            market=market,
            data_report=data_report,
            technical_report=technical_report,
            sentiment_report=sentiment_report,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.get("temperature", 0.1),
                max_tokens=self.llm_config.get("max_tokens", 1000),
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            decision = json.loads(raw)
            return self._format_decision(symbol, decision)
        except Exception as e:
            logger.error(f"LLM Strategy error for {symbol}: {e}")
            return f"HOLD | Confiance: 0% | Erreur LLM: {e}"

    def _format_decision(self, symbol: str, d: dict) -> str:
        action = d.get("action", "HOLD")
        confidence = d.get("confidence", 0)
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")

        signals = "\n".join(f"  ✓ {s}" for s in d.get("key_signals", []))
        risks = "\n".join(f"  ⚠ {r}" for r in d.get("risks", []))

        return (
            f"Décision LLM pour {symbol}\n"
            f"{'─'*50}\n"
            f"{emoji} ACTION: {action} | Confiance: {confidence:.0%}\n\n"
            f"Raisonnement:\n  {d.get('reasoning', 'N/A')}\n\n"
            f"Signaux clés:\n{signals}\n\n"
            f"Risques identifiés:\n{risks}\n\n"
            f"Zones:\n"
            f"  Entrée: {d.get('entry_zone', 'N/A')}\n"
            f"  Stop-Loss: {d.get('stop_loss_hint', 'N/A')}\n"
            f"  Take-Profit: {d.get('take_profit_hint', 'N/A')}\n"
            f"  Horizon: {d.get('timeframe', 'N/A')}"
        )


def create_strategy_agent(llm_config: dict) -> tuple[Agent, Task]:
    tool = LLMStrategyTool(llm_config=llm_config)

    agent = Agent(
        role="Chief Trading Strategist (LLM)",
        goal=(
            "Synthétiser tous les signaux (technique, sentiment, données) via raisonnement LLM "
            "pour produire des décisions de trading précises BUY/SELL/HOLD avec niveau de confiance"
        ),
        backstory=(
            "Tu es le stratégiste en chef d'un fonds quant. Tu reçois les rapports de tous les agents "
            "spécialisés et tu prends la décision finale de trading. Tu combines l'analyse technique, "
            "le sentiment de marché et la macro pour identifier les meilleures opportunités sur "
            "les marchés crypto, forex, actions et matières premières (or, pétrole)."
        ),
        tools=[tool],
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Pour chaque instrument {symbols}, analyse l'ensemble des rapports fournis par les agents "
            "Data, Technical et Sentiment. Prends une décision BUY/SELL/HOLD avec: "
            "1) Un niveau de confiance (0-100%), "
            "2) Un raisonnement clair et quantifié, "
            "3) Les niveaux d'entrée, stop-loss et take-profit recommandés, "
            "4) L'horizon temporel de la trade."
        ),
        expected_output=(
            "Une décision structurée par instrument: action (BUY/SELL/HOLD), confiance %, "
            "raisonnement, signaux convergents, risques, et niveaux de prix cibles."
        ),
        agent=agent,
        context=[],  # sera rempli dynamiquement avec les tâches précédentes
    )

    return agent, task
