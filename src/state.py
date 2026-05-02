"""
LangGraph shared state: grounding context, curricula, critique loop, and trace flags.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .schemas import GrammarCurriculum, VocabularyCurriculum


class GraphState(TypedDict, total=False):
    """
    State passed between nodes.

    Fields marked optional at type level may still be set by nodes during the run.
    """

    # Conversation / agent scratchpad (append-only via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Request parameters
    language: str
    cefr_level: str

    # Authoritative grounding (from read_authoritative_guidelines)
    grounding_context: str

    # Working curricula (validated Pydantic roots)
    vocabulary_curriculum: VocabularyCurriculum | None
    grammar_curriculum: GrammarCurriculum | None

    # Critique loop
    validation_feedback: str
    status: Literal["pending", "approved", "needs_revision"]
    validation_errors: list[str]
    revision_count: int
    max_revisions: int

    # Legacy trace (optional; research_node may still append paths)
    source_paths_used: list[str]


def initial_state(
    language: str,
    cefr_level: str,
    *,
    max_revisions: int = 3,
) -> GraphState:
    """Factory for a fresh graph invocation."""
    return GraphState(
        messages=[],
        language=language,
        cefr_level=cefr_level,
        grounding_context="",
        vocabulary_curriculum=None,
        grammar_curriculum=None,
        validation_feedback="",
        status="pending",
        validation_errors=[],
        revision_count=0,
        max_revisions=max_revisions,
        source_paths_used=[],
    )
