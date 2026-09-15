from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import AgentNodes


def build_agent_graph(
    tools: list,
    memory,
    risk_guard,
    deepseek_model: str = "deepseek-v4.1-flash",
    deepseek_api_key: str = "",
    max_iterations: int = 10,
) -> StateGraph:
    """Build the LangGraph agent with Observe → Reason → Act → Reflect loop."""
    nodes = AgentNodes(
        tools=tools,
        memory=memory,
        risk_guard=risk_guard,
        deepseek_model=deepseek_model,
        deepseek_api_key=deepseek_api_key,
        max_iterations=max_iterations,
    )

    graph = StateGraph(AgentState)

    graph.add_node("observe", nodes.observe_node)
    graph.add_node("reason", nodes.reason_node)
    graph.add_node("act", nodes.act_node)
    graph.add_node("reflect", nodes.reflect_node)

    graph.set_entry_point("observe")
    graph.add_edge("observe", "reason")
    graph.add_edge("reason", "act")
    graph.add_edge("act", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda s: s.get("_next", "observe"),
        {"observe": "observe", "END": END},
    )

    return graph.compile(checkpointer=MemorySaver())
