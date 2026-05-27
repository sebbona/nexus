"""CrewAI orchestration: coordination des 5 agents de trading."""

from __future__ import annotations

from crewai import Crew, Process
from loguru import logger

from agents import (
    create_data_agent,
    create_sentiment_agent,
    create_technical_agent,
    create_strategy_agent,
    create_risk_agent,
)
from data.fetcher import MarketDataFetcher
from data.news import NewsFetcher
from models.sentiment import FinBERTSentiment


class NexusTradingCrew:
    """
    Orchestration des 5 agents IA pour le trading multi-marché.
    Les agents s'exécutent en séquence: Data → Sentiment + Technical → Strategy → Risk.
    """

    def __init__(self, config: dict):
        self.config = config
        self._setup_shared_services()
        self._build_crew()

    def _setup_shared_services(self):
        logger.info("Initialisation des services partagés...")
        self.fetcher = MarketDataFetcher(self.config)
        self.news_fetcher = NewsFetcher(self.config)
        self.sentiment_model = FinBERTSentiment(
            device=self.config["models"]["sentiment"]["device"],
            batch_size=self.config["models"]["sentiment"]["batch_size"],
        )
        logger.success("Services initialisés")

    def _build_crew(self):
        symbols = self._get_all_symbols()
        symbols_str = ", ".join(symbols)

        data_agent, data_task = create_data_agent(self.fetcher)
        sentiment_agent, sentiment_task = create_sentiment_agent(
            self.news_fetcher, self.sentiment_model
        )
        technical_agent, technical_task = create_technical_agent(self.fetcher)
        strategy_agent, strategy_task = create_strategy_agent(
            self.config["models"]["llm"]
        )
        risk_agent, risk_task = create_risk_agent(self.config["risk"])

        # Injection des symboles dans les descriptions des tâches
        for task in [data_task, sentiment_task, technical_task, strategy_task, risk_task]:
            task.description = task.description.format(symbols=symbols_str)

        # Le stratégiste a besoin des outputs des autres agents
        strategy_task.context = [data_task, sentiment_task, technical_task]
        risk_task.context = [strategy_task]

        self.crew = Crew(
            agents=[data_agent, sentiment_agent, technical_agent, strategy_agent, risk_agent],
            tasks=[data_task, sentiment_task, technical_task, strategy_task, risk_task],
            process=Process.sequential,
            verbose=True,
            memory=False,
        )

        self.agents_map = {
            "data": data_agent,
            "sentiment": sentiment_agent,
            "technical": technical_agent,
            "strategy": strategy_agent,
            "risk": risk_agent,
        }

    def run_analysis(self) -> str:
        """Lance l'analyse complète multi-agents et retourne le rapport final."""
        logger.info("Lancement de l'analyse NexusTrader...")
        result = self.crew.kickoff()
        logger.success("Analyse terminée")
        return str(result)

    def run_quick_analysis(self, symbol: str, market: str) -> str:
        """Analyse rapide d'un seul instrument."""
        symbols_str = f"{symbol} ({market})"
        data_agent, data_task = create_data_agent(self.fetcher)
        technical_agent, technical_task = create_technical_agent(self.fetcher)
        strategy_agent, strategy_task = create_strategy_agent(self.config["models"]["llm"])
        risk_agent, risk_task = create_risk_agent(self.config["risk"])

        for task in [data_task, technical_task, strategy_task, risk_task]:
            task.description = task.description.format(symbols=symbols_str)

        strategy_task.context = [data_task, technical_task]
        risk_task.context = [strategy_task]

        crew = Crew(
            agents=[data_agent, technical_agent, strategy_agent, risk_agent],
            tasks=[data_task, technical_task, strategy_task, risk_task],
            process=Process.sequential,
            verbose=False,
        )
        return str(crew.kickoff())

    def _get_all_symbols(self) -> list[str]:
        symbols = []
        for market, cfg in self.config["markets"].items():
            if cfg["enabled"]:
                symbols.extend(cfg["symbols"])
        return symbols
