import pytest
import os
import tempfile

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory
from src.memory.store import AgentMemory


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_journal.db")


@pytest.fixture
def episodic_memory(tmp_db):
    return EpisodicMemory(db_path=tmp_db)


class TestEpisodicMemory:
    def test_log_and_retrieve_trades(self, episodic_memory):
        episodic_memory.log_trade({"symbol": "BTCUSDT", "decision": "BUY", "pnl": 100, "reasoning": "trend"})
        trades = episodic_memory.get_recent_trades(limit=5)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTCUSDT"
        assert trades[0]["pnl"] == 100

    def test_performance_stats_empty(self, episodic_memory):
        stats = episodic_memory.get_performance_stats()
        assert stats["total"] == 0
        assert stats["win_rate"] == 0

    def test_performance_stats_with_trades(self, episodic_memory):
        episodic_memory.log_trade({"symbol": "BTCUSDT", "decision": "BUY", "pnl": 100, "reasoning": "t1", "outcome": "win"})
        episodic_memory.log_trade({"symbol": "ETHUSDT", "decision": "SELL", "pnl": -50, "reasoning": "t2", "outcome": "loss"})
        stats = episodic_memory.get_performance_stats()
        assert stats["total"] == 2
        assert stats["win_rate"] == 0.5
        assert stats["avg_pnl"] == 25

    def test_log_error(self, episodic_memory):
        episodic_memory.log_error("Test error")
        errors = episodic_memory.get_all_errors()
        assert len(errors) == 1
        assert "Test error" in errors[0]["error"]

    def test_get_recent_trades_limit(self, episodic_memory):
        for i in range(5):
            episodic_memory.log_trade({"symbol": f"SYM{i}", "decision": "BUY", "pnl": i * 10})
        trades = episodic_memory.get_recent_trades(limit=3)
        assert len(trades) == 3
        assert trades[0]["symbol"] == "SYM4"  # Most recent first


class TestWorkingMemory:
    def test_add_and_get_context(self):
        wm = WorkingMemory(max_context_turns=5)
        wm.add_turn("thought1", "action1", "result1")
        wm.add_turn("thought2", "action2", "result2")
        context = wm.get_context()
        assert "thought1" in context
        assert "action2" in context

    def test_max_context_limit(self):
        wm = WorkingMemory(max_context_turns=3)
        for i in range(5):
            wm.add_turn(f"thought{i}", f"action{i}", f"result{i}")
        assert len(wm.context_turns) == 3
        assert wm.context_turns[0]["thought"] == "thought2"  # First two removed

    def test_summarize(self):
        wm = WorkingMemory(max_context_turns=10)
        wm.add_turn("thought1", "action1", "result1")
        summary = wm.summarize()
        assert "thought1" in summary

    def test_clear(self):
        wm = WorkingMemory()
        wm.add_turn("thought1", "action1", "result1")
        wm.clear()
        assert wm.get_turn_count() == 0

    def test_turn_count(self):
        wm = WorkingMemory()
        assert wm.get_turn_count() == 0
        wm.add_turn("t", "a", "r")
        assert wm.get_turn_count() == 1


class TestAgentMemory:
    def test_log_trade(self, tmp_db):
        mem = AgentMemory(db_path=tmp_db)
        mem.log_trade({"symbol": "BTCUSDT", "decision": "BUY", "pnl": 100})
        trades = mem.get_recent_trades()
        assert len(trades) == 1

    def test_add_turn(self, tmp_db):
        mem = AgentMemory(db_path=tmp_db)
        mem.add_turn("thought", "action", "result")
        assert mem.get_turn_count() == 1
        assert "thought" in mem.get_context()

    def test_log_error(self, tmp_db):
        mem = AgentMemory(db_path=tmp_db)
        mem.log_error("test")
        errors = mem.get_errors()
        assert len(errors) == 1

    def test_performance_stats(self, tmp_db):
        mem = AgentMemory(db_path=tmp_db)
        assert mem.get_performance_stats()["total"] == 0
        mem.log_trade({"symbol": "BTC", "decision": "BUY", "pnl": 50, "outcome": "win"})
        stats = mem.get_performance_stats()
        assert stats["total"] == 1


class TestSemanticMemory:
    def test_add_and_search_strategy(self, tmp_path):
        # ChromaDB needs a persistent path for in-process testing
        import chromadb
        client = chromadb.Client()  # In-memory for testing
        collection = client.get_or_create_collection("test_strategies")
        collection.add(documents=["buy on RSI < 30"], ids=["rsi_buy"])
        results = collection.query(query_texts=["RSI strategy"], n_results=1)
        assert len(results["ids"][0]) == 1