"""
LangGraph nodes: research (tool-grounded retrieval), structured generation, pedagogical critique.

Nodes return partial ``GraphState`` updates. Generation uses ``ChatGoogleGenerativeAI`` with
``.with_structured_output()`` for ``VocabularyCurriculum`` and ``GrammarCurriculum`` separately.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .config import get_settings
from .schemas import (
    CEFRLevel,
    CurriculumMetadata,
    GrammarCurriculum,
    VocabularyCurriculum,
)
from .state import GraphState
from .tools import read_authoritative_guidelines, read_authoritative_guidelines_text

GUIDELINES_SOURCE_REF = "data/input/cefr_guidelines.md"


class AgentNode(Protocol):
    """Protocol for typed graph nodes."""

    def __call__(self, state: GraphState) -> dict[str, Any]: ...


class CritiqueDecision(BaseModel):
    """Structured output for the pedagogical critic."""

    status: Literal["approved", "needs_revision"] = Field(
        ...,
        description='Whether outputs are grounded enough, or should be regenerated.',
    )
    validation_feedback: str = Field(
        ...,
        description='Concrete feedback for the generator if needs_revision; else brief approval note.',
    )


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_cefr(level: str) -> CEFRLevel:
    return CEFRLevel(level.strip().upper())


def _make_llm(*, temperature: float = 0.15) -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_model,
        google_api_key=s.google_api_key or None,
        temperature=temperature,
    )


def _stamp_vocabulary(meta_language: str, level: CEFRLevel, voc: VocabularyCurriculum) -> VocabularyCurriculum:
    md = CurriculumMetadata(
        generated_at_iso=_utc_iso_z(),
        pipeline_version=voc.metadata.pipeline_version,
        language=meta_language,
        cefr_target=level,
        grounding_sources=sorted(
            set(voc.metadata.grounding_sources) | {GUIDELINES_SOURCE_REF},
        ),
    )
    return voc.model_copy(update={"metadata": md})


def _stamp_grammar(meta_language: str, level: CEFRLevel, gr: GrammarCurriculum) -> GrammarCurriculum:
    md = CurriculumMetadata(
        generated_at_iso=_utc_iso_z(),
        pipeline_version=gr.metadata.pipeline_version,
        language=meta_language,
        cefr_target=level,
        grounding_sources=sorted(
            set(gr.metadata.grounding_sources) | {GUIDELINES_SOURCE_REF},
        ),
    )
    return gr.model_copy(update={"metadata": md})


def _tool_call_parts(tc: Any) -> tuple[str, dict[str, Any], str]:
    """Normalize LangChain / provider tool_call objects."""
    if isinstance(tc, dict):
        raw_args = tc.get("args", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        return (
            str(tc.get("name", "")),
            raw_args if isinstance(raw_args, dict) else {},
            str(tc.get("id", "tool-call")),
        )
    name = str(getattr(tc, "name", "") or "")
    raw_args = getattr(tc, "args", {}) or {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    tid = getattr(tc, "id", None)
    return name, raw_args, str(tid if tid is not None else "tool-call")


def _minimal_curricula(language: str, level_str: str) -> tuple[VocabularyCurriculum, GrammarCurriculum]:
    """Schema-valid empty curricula for offline / missing-key runs."""
    level = _parse_cefr(level_str)
    meta = CurriculumMetadata(
        generated_at_iso=_utc_iso_z(),
        pipeline_version="0.1.0",
        language=language,
        cefr_target=level,
        grounding_sources=[GUIDELINES_SOURCE_REF],
    )
    return (
        VocabularyCurriculum(metadata=meta, vocabulary_sections=[]),
        GrammarCurriculum(metadata=meta.model_copy(), grammar_sections=[]),
    )


def research_node(state: GraphState) -> dict[str, Any]:
    """
    Bind ``read_authoritative_guidelines`` to Gemini and retrieve grounding text.

    Executes the tool with the state's ``language`` and ``cefr_level``. If the model
    does not emit a tool call, falls back to a direct tool invocation so grounding
    stays deterministic.
    """
    language = state.get("language") or "English"
    level = state.get("cefr_level") or "A1"
    settings = get_settings()

    # Deterministic path when no API key: still load authoritative file via tool implementation.
    if not settings.google_api_key:
        text = read_authoritative_guidelines_text(language, level)
        return {
            "grounding_context": text,
            "source_paths_used": [GUIDELINES_SOURCE_REF],
            "messages": [
                AIMessage(
                    content=(
                        "[research_node] GOOGLE_API_KEY unset; loaded grounding via "
                        "read_authoritative_guidelines without LLM tool routing."
                    ),
                ),
            ],
        }

    llm = _make_llm(temperature=0.0)
    llm_tools = llm.bind_tools([read_authoritative_guidelines])
    prompt = HumanMessage(
        content=(
            "You must call the tool `read_authoritative_guidelines` exactly once using these arguments:\n"
            f'  language: "{language}"\n'
            f'  level: "{level}"\n'
            "Do not paraphrase CEFR content yourself; retrieval must come from the tool."
        ),
    )
    ai = llm_tools.invoke([prompt])
    grounding = ""
    tool_messages: list[ToolMessage] = []

    tool_calls = getattr(ai, "tool_calls", None) or []
    for tc in tool_calls:
        name, args, tid = _tool_call_parts(tc)
        if name != "read_authoritative_guidelines":
            continue
        lang_arg = args.get("language", language)
        lev_arg = args.get("level", level)
        grounding = read_authoritative_guidelines.invoke({"language": lang_arg, "level": lev_arg})
        if not isinstance(grounding, str):
            grounding = str(grounding)
        tool_messages.append(
            ToolMessage(content=grounding[:120_000], tool_call_id=tid),
        )

    if not grounding.strip():
        grounding = read_authoritative_guidelines_text(language, level)

    out_msgs: list[Any] = [prompt, ai]
    out_msgs.extend(tool_messages)
    return {
        "grounding_context": grounding,
        "source_paths_used": [GUIDELINES_SOURCE_REF],
        "messages": out_msgs,
    }


def generate_node(state: GraphState) -> dict[str, Any]:
    """
    Generate ``VocabularyCurriculum`` then ``GrammarCurriculum`` using structured outputs.

    Prompts require alignment with ``grounding_context`` and ``source_refs`` pointing at
    ``data/input/cefr_guidelines.md`` where claims derive from that text.
    """
    language = state.get("language") or "English"
    level_str = state.get("cefr_level") or "A1"
    grounding = (state.get("grounding_context") or "").strip()
    feedback = (state.get("validation_feedback") or "").strip()
    settings = get_settings()

    try:
        level_enum = _parse_cefr(level_str)
    except ValueError:
        return {
            "vocabulary_curriculum": None,
            "grammar_curriculum": None,
            "validation_feedback": f"Invalid CEFR level: {level_str!r}",
            "validation_errors": [f"Invalid CEFR level: {level_str!r}"],
            "messages": [
                AIMessage(content=f"[generate_node] Invalid level {level_str!r}."),
            ],
        }

    if not settings.google_api_key:
        voc, gr = _minimal_curricula(language, level_str)
        return {
            "vocabulary_curriculum": voc,
            "grammar_curriculum": gr,
            "messages": [
                AIMessage(
                    content="[generate_node] GOOGLE_API_KEY missing; emitted minimal empty curricula.",
                ),
            ],
            "validation_errors": ["GOOGLE_API_KEY not set; empty curricula for wiring."],
        }

    sys_shared = (
        "You are an expert language curriculum designer. You MUST only include vocabulary themes and "
        "grammar structures that are justified by the AUTHORITATIVE GROUNDING excerpt below. "
        f"Every vocabulary item and grammar rule MUST include '{GUIDELINES_SOURCE_REF}' in source_refs "
        "(and/or chunk identifiers from that file) when the content is supported by the grounding text. "
        "Do not invent advanced structures beyond the target CEFR band relative to the grounding. "
        "If the grounding does not support an item, omit it rather than hallucinating."
    )
    revision_block = (
        f"\n\nPrior critique (must address if non-empty):\n{feedback}\n"
        if feedback
        else ""
    )
    human_grounding = (
        f"Target language: {language}\n"
        f"CEFR target level: {level_str}\n\n"
        f"AUTHORITATIVE GROUNDING:\n{grounding}\n"
        f"{revision_block}"
    )

    llm_base = _make_llm(temperature=0.2)
    errors: list[str] = []

    try:
        llm_v = llm_base.with_structured_output(VocabularyCurriculum)
        voc_raw = llm_v.invoke(
            [
                SystemMessage(content=sys_shared),
                HumanMessage(
                    content=human_grounding
                    + "\nProduce the vocabulary curriculum (sections and items) for this level.",
                ),
            ],
        )
        if not isinstance(voc_raw, VocabularyCurriculum):
            raise TypeError("Structured output was not VocabularyCurriculum")
        vocabulary = _stamp_vocabulary(language, level_enum, voc_raw)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Vocabulary generation failed: {exc}")
        return {
            "vocabulary_curriculum": None,
            "grammar_curriculum": None,
            "validation_feedback": errors[0],
            "validation_errors": errors,
            "messages": [AIMessage(content=f"[generate_node] {errors[0]}")],
        }

    try:
        llm_g = llm_base.with_structured_output(GrammarCurriculum)
        gr_raw = llm_g.invoke(
            [
                SystemMessage(content=sys_shared),
                HumanMessage(
                    content=human_grounding
                    + "\nProduce the grammar curriculum (sections and rules) for this level. "
                    "Ensure rule_id values are unique.",
                ),
            ],
        )
        if not isinstance(gr_raw, GrammarCurriculum):
            raise TypeError("Structured output was not GrammarCurriculum")
        grammar = _stamp_grammar(language, level_enum, gr_raw)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Grammar generation failed: {exc}")
        return {
            "vocabulary_curriculum": vocabulary,
            "grammar_curriculum": None,
            "validation_feedback": errors[0],
            "validation_errors": errors,
            "messages": [AIMessage(content=f"[generate_node] {errors[0]}")],
        }

    return {
        "vocabulary_curriculum": vocabulary,
        "grammar_curriculum": grammar,
        "validation_errors": [],
        "messages": [
            AIMessage(
                content=(
                    "[generate_node] Structured generation completed for vocabulary and grammar roots."
                ),
            ),
        ],
    }


def _critique_with_llm(
    *,
    grounding: str,
    language: str,
    level_str: str,
    voc: VocabularyCurriculum,
    gr: GrammarCurriculum,
) -> CritiqueDecision:
    llm = _make_llm(temperature=0.0).with_structured_output(CritiqueDecision)
    summary = (
        f"Vocabulary sections: {len(voc.vocabulary_sections)}, "
        f"Grammar sections: {len(gr.grammar_sections)}.\n"
        f"Vocabulary JSON (truncated):\n{voc.model_dump_json()[:6000]}\n\n"
        f"Grammar JSON (truncated):\n{gr.model_dump_json()[:6000]}\n"
    )
    msg = HumanMessage(
        content=(
            f"You are a pedagogical auditor. Target language={language}, CEFR={level_str}.\n"
            f"Compare the curriculum drafts to the AUTHORITATIVE GROUNDING below. "
            f"Reject (needs_revision) if you detect hallucinated grammar/vocabulary not supported by the "
            f"grounding, or content clearly above the target CEFR band. "
            f"Approve only if content is plausibly grounded and level-appropriate.\n\n"
            f"AUTHORITATIVE GROUNDING:\n{grounding[:14_000]}\n\n"
            f"CURRICULUM SUMMARY / EXCERPTS:\n{summary}"
        ),
    )
    out = llm.invoke([msg])
    if not isinstance(out, CritiqueDecision):
        raise TypeError("Critique structured output invalid")
    return out


def _critique_heuristic(
    grounding: str,
    voc: VocabularyCurriculum | None,
    gr: GrammarCurriculum | None,
) -> CritiqueDecision:
    if voc is None or gr is None:
        return CritiqueDecision(
            status="needs_revision",
            validation_feedback="Missing vocabulary or grammar curriculum object.",
        )
    if "[Grounding error]" in grounding:
        return CritiqueDecision(
            status="needs_revision",
            validation_feedback="Grounding file unavailable; cannot approve content as authoritative.",
        )
    return CritiqueDecision(
        status="approved",
        validation_feedback=(
            "Heuristic pass: both Pydantic models present. "
            "Set GOOGLE_API_KEY for LLM-based hallucination audit against grounding."
        ),
    )


def critique_node(state: GraphState) -> dict[str, Any]:
    """
    Evaluate drafts against ``grounding_context``; set ``status`` and ``validation_feedback``.
    """
    grounding = state.get("grounding_context") or ""
    language = state.get("language") or "English"
    level_str = state.get("cefr_level") or "A1"
    voc = state.get("vocabulary_curriculum")
    gr = state.get("grammar_curriculum")
    settings = get_settings()
    prev_rev = int(state.get("revision_count") or 0)

    try:
        if settings.google_api_key and isinstance(voc, VocabularyCurriculum) and isinstance(gr, GrammarCurriculum):
            decision = _critique_with_llm(
                grounding=grounding,
                language=language,
                level_str=level_str,
                voc=voc,
                gr=gr,
            )
        else:
            decision = _critique_heuristic(grounding, voc, gr)
    except Exception as exc:  # noqa: BLE001
        decision = CritiqueDecision(
            status="needs_revision",
            validation_feedback=f"Critique failed: {exc}",
        )

    if decision.status == "approved":
        return {
            "status": "approved",
            "validation_feedback": decision.validation_feedback,
            "validation_errors": [],
            "messages": [
                AIMessage(
                    content=f"[critique_node] Approved. Notes: {decision.validation_feedback[:500]}",
                ),
            ],
        }

    new_rev = prev_rev + 1
    return {
        "status": "needs_revision",
        "validation_feedback": decision.validation_feedback,
        "validation_errors": [decision.validation_feedback],
        "revision_count": new_rev,
        "messages": [
            AIMessage(
                content=(
                    f"[critique_node] Needs revision (revision_count={new_rev}). "
                    f"{decision.validation_feedback[:800]}"
                ),
            ),
        ],
    }


def build_default_agents() -> tuple[
    Callable[[GraphState], dict[str, Any]],
    Callable[[GraphState], dict[str, Any]],
    Callable[[GraphState], dict[str, Any]],
]:
    """Backward-compatible factory returning the three node callables."""
    return research_node, generate_node, critique_node
