"""
Compile the LangGraph ``StateGraph`` with research → generate → critique and revision cycles.

Flow:
    START → research_node → generate_node → critique_node → (END | generate_node)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import critique_node, generate_node, research_node
from .state import GraphState


def route_after_critique(state: GraphState) -> str:
    """Route to END when approved or revision budget exhausted; else regenerate."""
    if state.get("status") == "approved":
        return END
    max_r = int(state.get("max_revisions") or 3)
    rev = int(state.get("revision_count") or 0)
    if state.get("status") == "needs_revision" and rev < max_r:
        return "generate_node"
    return END


def compile_curriculum_graph() -> Any:
    """
    Build and compile the graph.

    Returns a compiled graph with ``invoke`` / ``ainvoke``.
    """
    graph = StateGraph(GraphState)

    graph.add_node("research_node", research_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("critique_node", critique_node)

    graph.add_edge(START, "research_node")
    graph.add_edge("research_node", "generate_node")
    graph.add_edge("generate_node", "critique_node")
    graph.add_conditional_edges(
        "critique_node",
        route_after_critique,
        {
            "generate_node": "generate_node",
            END: END,
        },
    )

    return graph.compile()
