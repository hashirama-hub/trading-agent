import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from langgraph.graph import END
from src.agent.state import AgentState, PortfolioState, MarketSnapshot
from src.agent.nodes import AgentNodes
from src.agent.graph import build_agent_graph
from src.memory.store import AgentMemory
from src.risk.guard import RiskGuard, RiskDecision
from src.tools.binance_client import OrderRequest, PortfolioState as BSPortfolioState


@pytest.fixture
def agent_nodes():
    risk_guard = RiskGuard(
        BSPortfolioState(equity=10000, positions=[], daily_pnl=0)
    )
    memory = AgentMemory(db_path="/tmp/test_agent_memory.db")
    nodes = AgentNodes(
        tools=[],
        memory=memory,
        risk_guard=risk_guard,
        max_iterations=2,
    )
    # Mock LLM via property
    nodes._llm = MagicMock()
    return nodes


class TestObserveNode:
    def test_observe_populates_market_data(self, agent_nodes):
        state = AgentState(
            market_data={"BTCUSDT": {"close": 100000}},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision=None,
            errors=[],
        )
        result = agent_nodes.observe_node(state)
        assert "BTCUSDT" in result["market_data"]
        assert result["iteration"] == 1

    def test_observe_increments_iteration(self, agent_nodes):
        state = AgentState(
            market_data={},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=5,
            max_iterations=10,
            last_decision=None,
            errors=[],
        )
        result = agent_nodes.observe_node(state)
        assert result["iteration"] == 6


class TestReasonNode:
    def test_reason_node_calls_llm(self, agent_nodes):
        mock_response = MagicMock()
        mock_response.content = "Analyzing market..."
        mock_response.tool_calls = None
        agent_nodes.llm.invoke.return_value = mock_response

        state = AgentState(
            market_data={"BTCUSDT": {"close": 100000}},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision=None,
            errors=[],
        )
        result = agent_nodes.reason_node(state)
        assert result["last_decision"] is not None
        assert result["last_decision"]["type"] == "reasoning"
        assert "Analyzing market" in result["last_decision"]["thought"]

    def test_reason_node_with_tool_call(self, agent_nodes):
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [
            {
                "name": "get_market_data",
                "args": {"symbol": "BTCUSDT", "timeframe": "1h"},
            }
        ]
        agent_nodes.llm.invoke.return_value = mock_response

        state = AgentState(
            market_data={},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision=None,
            errors=[],
        )
        result = agent_nodes.reason_node(state)
        assert result["last_decision"]["type"] == "action"
        assert result["last_decision"]["action"] == "get_market_data"
        assert result["last_decision"]["params"]["symbol"] == "BTCUSDT"


class TestActNode:
    def test_act_node_approves_valid_order(self, agent_nodes):
        state = AgentState(
            market_data={"BTCUSDT": {"close": 100000}},
            portfolio={"equity": 10000, "available_margin": 5000, "positions": [], "daily_pnl": 0, "max_drawdown_today": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision={
                "type": "action",
                "action": "place_order",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "qty": 0.01,
                    "price": 100000,
                    "stop_loss": 99000,
                    "take_profit": 102000,
                },
                "thought": "Trend detected, buying",
            },
            errors=[],
        )
        result = agent_nodes.act_node(state)
        assert len(state["errors"]) == 0  # No errors

    def test_act_node_rejects_no_stop_loss(self, agent_nodes):
        state = AgentState(
            market_data={"BTCUSDT": {"close": 100000}},
            portfolio={"equity": 10000, "available_margin": 5000, "positions": [], "daily_pnl": 0, "max_drawdown_today": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision={
                "type": "action",
                "action": "place_order",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "qty": 0.01,
                    "price": 100000,
                },
                "thought": "No stop loss",
            },
            errors=[],
        )
        result = agent_nodes.act_node(state)
        assert any("Stop-loss" in e for e in state["errors"])

    def test_act_node_modifies_oversized_position(self, agent_nodes):
        state = AgentState(
            market_data={"BTCUSDT": {"close": 100000}},
            portfolio={"equity": 10000, "available_margin": 5000, "positions": [], "daily_pnl": 0, "max_drawdown_today": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision={
                "type": "action",
                "action": "place_order",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "qty": 100.0,
                    "price": 100000,
                    "stop_loss": 99000,
                    "take_profit": 102000,
                },
                "thought": "Oversized position",
            },
            errors=[],
        )
        agent_nodes.act_node(state)
        # Oversized should be modified to within limits, not error
        assert len(state["errors"]) == 0


class TestReflectNode:
    def test_reflect_loops_to_observe(self, agent_nodes):
        state = AgentState(
            market_data={},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=2,
            last_decision=None,
            errors=[],
        )
        result = agent_nodes.reflect_node(state)
        assert result["_next"] == "observe"

    def test_reflect_ends_at_max_iterations(self, agent_nodes):
        state = AgentState(
            market_data={},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=2,
            max_iterations=2,
            last_decision=None,
            errors=[],
        )
        result = agent_nodes.reflect_node(state)
        assert result["_next"] == END


class TestGraphBuilder:
    def test_graph_builds_correctly(self):
        memory = AgentMemory(db_path="/tmp/test_graph_memory.db")
        risk_guard = RiskGuard(
            BSPortfolioState(equity=10000, positions=[], daily_pnl=0)
        )
        nodes = AgentNodes(
            tools=[], memory=memory, risk_guard=risk_guard, max_iterations=2
        )
        nodes._llm = MagicMock()
        graph = build_agent_graph(
            tools=[],
            memory=memory,
            risk_guard=risk_guard,
            max_iterations=2,
        )
        assert graph is not None

    def test_graph_has_all_nodes(self):
        memory = AgentMemory(db_path="/tmp/test_graph_nodes.db")
        risk_guard = RiskGuard(
            BSPortfolioState(equity=10000, positions=[], daily_pnl=0)
        )
        nodes = AgentNodes(
            tools=[], memory=memory, risk_guard=risk_guard, max_iterations=2
        )
        nodes._llm = MagicMock()
        graph = build_agent_graph(
            tools=[],
            memory=memory,
            risk_guard=risk_guard,
            max_iterations=2,
        )
        assert hasattr(graph, "nodes")


class TestAgentIntegration:
    def test_full_agent_loop(self):
        """Test complete observe → reason → act → reflect cycle."""
        memory = AgentMemory(db_path="/tmp/test_integration.db")
        risk_guard = RiskGuard(
            BSPortfolioState(equity=10000, positions=[], daily_pnl=0)
        )

        nodes = AgentNodes(
            tools=[],
            memory=memory,
            risk_guard=risk_guard,
            max_iterations=1,
        )
        nodes._llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Market looks stable."
        mock_response.tool_calls = None
        nodes._llm.invoke.return_value = mock_response

        state = AgentState(
            market_data={},
            portfolio={"equity": 10000, "positions": [], "daily_pnl": 0},
            memory={"working": "", "episodic_trades": [], "strategies": []},
            iteration=0,
            max_iterations=1,
            last_decision=None,
            errors=[],
        )

        state = nodes.observe_node(state)
        state = nodes.reason_node(state)
        state = nodes.act_node(state)
        state = nodes.reflect_node(state)

        assert state["iteration"] == 1
        assert state["last_decision"] is not None
        assert memory.get_turn_count() >= 1