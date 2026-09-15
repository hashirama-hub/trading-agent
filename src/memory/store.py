from typing import List, Dict, Optional

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory


class AgentMemory:
    def __init__(self, db_path: str = "data/trade_journal.db", chroma_host: str = "chromadb", chroma_port: int = 8000):
        self.episodic = EpisodicMemory(db_path=db_path)
        self.semantic = SemanticMemory(host=chroma_host, port=chroma_port)
        self.working = WorkingMemory()

    def log_trade(self, trade: Dict):
        self.episodic.log_trade(trade)

    def log_error(self, error: str):
        self.episodic.log_error(error)

    def add_turn(self, thought: str, action: str, result: str):
        self.working.add_turn(thought, action, result)

    def add_strategy(self, name: str, content: str, metadata: Dict = None):
        self.semantic.add_strategy(name, content, metadata)

    def search_strategies(self, query: str, n_results: int = 5) -> list:
        return self.semantic.search(query, n_results)

    def get_performance_stats(self) -> dict:
        return self.episodic.get_performance_stats()

    def get_performance_by_regime(self) -> dict:
        return self.episodic.get_performance_by_regime()

    def get_performance_by_confidence(self) -> dict:
        return self.episodic.get_performance_by_confidence()

    def get_recent_trades(self, limit: int = 20) -> list:
        return self.episodic.get_recent_trades(limit)

    def get_pending_trades(self) -> list:
        return self.episodic.get_pending_trades()

    def update_trade_outcome(self, trade_id: int, outcome: str, actual_pnl: float,
                             exit_price: float = None, hold_time_seconds: int = None):
        self.episodic.update_trade_outcome(trade_id, outcome, actual_pnl, exit_price, hold_time_seconds)

    def get_winning_patterns(self, limit: int = 10) -> list:
        return self.episodic.get_winning_patterns(limit)

    def save_performance_snapshot(self, stats: dict):
        self.episodic.save_performance_snapshot(stats)

    def get_errors(self) -> list:
        return self.episodic.get_all_errors()

    def get_context(self) -> str:
        return self.working.get_context()

    def get_turn_count(self) -> int:
        return self.working.get_turn_count()
