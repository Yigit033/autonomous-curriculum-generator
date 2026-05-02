"""
Grounding tools: load CEFR rules and reference material from ``data/input``.

Keep retrieval deterministic (filesystem reads, optional chunking) to reduce hallucinations:
agents should cite ``source_refs`` that map to paths or chunk IDs produced here.

The ``read_authoritative_guidelines`` LangChain tool reads the local authoritative markdown
``cefr_guidelines.md`` so curriculum claims can be traced to file-backed content rather than
model priors alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.tools import tool

from .config import Settings, get_settings

CEFR_GUIDELINES_FILENAME = "cefr_guidelines.md"
# Return full file instead of a filtered slice when under this size (mock KB stays small).
_FULL_DOCUMENT_CHAR_THRESHOLD = 50_000


@dataclass(frozen=True, slots=True)
class GroundingChunk:
    """A slice of grounded text with a stable identifier for citation."""

    chunk_id: str
    source_path: str
    text: str


def _default_settings() -> Settings:
    return get_settings()


def _guidelines_path() -> Path:
    return _default_settings().data_input_dir / CEFR_GUIDELINES_FILENAME


def _normalize_language_key(language: str) -> str | None:
    """Map free-text language to internal keys used in ``cefr_guidelines.md``."""
    raw = language.strip().lower()
    if raw in ("english", "en", "eng"):
        return "english"
    if raw in ("spanish", "es", "esp", "español", "espanol", "castilian"):
        return "spanish"
    return None


def _normalize_cefr_level(level: str) -> str | None:
    lev = level.strip().upper()
    if lev in ("A1", "A2", "B1"):
        return lev
    return None


def _extract_language_section(markdown: str, lang_key: str) -> str | None:
    """Return body text under ``## English`` or ``## Spanish`` until the next ``## `` heading."""
    titles = {"english": "English", "spanish": "Spanish"}
    title = titles.get(lang_key)
    if not title:
        return None
    start_marker = f"## {title}"
    lines = markdown.splitlines()
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == start_marker:
            start_idx = i + 1
            break
    if start_idx is None:
        return None
    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        stripped = lines[j]
        if stripped.startswith("## ") and stripped.strip() != start_marker:
            end_idx = j
            break
    return "\n".join(lines[start_idx:end_idx]).strip()


def _extract_level_slice(language_block: str, level: str) -> str | None:
    """Return text under ``### A1`` / ``### A2`` / ``### B1`` within a language section."""
    marker = f"### {level}"
    lines = language_block.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == marker:
            chunk: list[str] = []
            for sub in lines[i + 1 :]:
                if sub.startswith("### ") and sub.strip() != marker:
                    break
                chunk.append(sub)
            text = "\n".join(chunk).strip()
            return text if text else None
    return None


def _build_grounded_payload(
    *,
    relative_path: str,
    language: str,
    level: str,
    body: str,
) -> str:
    header = (
        f"[Authoritative source: {relative_path} | language={language!r} | level={level!r}]\n"
        f"[Instruction: align curriculum claims with this excerpt; cite this path in source_refs.]\n\n"
    )
    return header + body


def _read_authoritative_guidelines_core(language: str, level: str) -> str:
    """Implementation shared by the LangChain tool and non-tool callers."""
    lang_key = _normalize_language_key(language)
    lev = _normalize_cefr_level(level)
    rel_path = f"data/input/{CEFR_GUIDELINES_FILENAME}"
    path = _guidelines_path()

    try:
        full_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            f"[Grounding error] Authoritative file not found: {path}. "
            f"Restore `{CEFR_GUIDELINES_FILENAME}` under `data/input/` to enable RAG-style grounding."
        )
    except OSError as exc:
        return f"[Grounding error] Could not read {path}: {exc}"

    # Unknown language/level: still return useful grounding (whole doc if concise).
    if lang_key is None or lev is None:
        note = (
            f"[Grounding notice] Could not map language={language!r} or level={level!r} "
            f"to a filtered section (supported languages: English, Spanish; levels: A1, A2, B1). "
            f"Returning full reference below when within size limits.\n\n"
        )
        if len(full_text) <= _FULL_DOCUMENT_CHAR_THRESHOLD:
            return note + full_text
        return note + full_text[:_FULL_DOCUMENT_CHAR_THRESHOLD] + "\n\n[…truncated…]"

    lang_body = _extract_language_section(full_text, lang_key)
    if not lang_body:
        if len(full_text) <= _FULL_DOCUMENT_CHAR_THRESHOLD:
            return (
                f"[Grounding notice] Language section not found for {language!r}; "
                f"returning full `{CEFR_GUIDELINES_FILENAME}`.\n\n" + full_text
            )
        return full_text[:_FULL_DOCUMENT_CHAR_THRESHOLD] + "\n\n[…truncated…]"

    level_slice = _extract_level_slice(lang_body, lev)
    if level_slice:
        return _build_grounded_payload(
            relative_path=rel_path,
            language=language.strip(),
            level=lev,
            body=level_slice,
        )

    if len(full_text) <= _FULL_DOCUMENT_CHAR_THRESHOLD:
        return (
            f"[Grounding notice] Level subsection `{lev}` not found under this language block; "
            f"returning full `{CEFR_GUIDELINES_FILENAME}`.\n\n" + full_text
        )
    return (
        f"[Grounding notice] Level `{lev}` not found; returning truncated full document.\n\n"
        + full_text[:_FULL_DOCUMENT_CHAR_THRESHOLD]
        + "\n\n[…truncated…]"
    )


@tool
def read_authoritative_guidelines(language: str, level: str) -> str:
    """Load CEFR grounding text from the local knowledge base (not from model priors).

    Reads ``data/input/cefr_guidelines.md`` and returns the section for the requested
    language and level when possible. This satisfies the product requirement that curriculum
    content be **grounded in authoritative local sources** rather than relying on LLM memory alone.

    Use before asserting vocabulary themes or grammar structures for English or Spanish at A1, A2, or B1.

    Args:
        language: Target language (e.g. ``English``, ``en``, ``Spanish``, ``es``).
        level: CEFR band (e.g. ``A1``, ``A2``, ``B1``).

    Returns:
        Filtered markdown excerpt, or the full guidelines document if filtering is unavailable
        and the file is small enough for the context window; or an error message string if the file is missing.
    """
    return _read_authoritative_guidelines_core(language, level)


def read_authoritative_guidelines_text(language: str, level: str) -> str:
    """Same behavior as the ``read_authoritative_guidelines`` tool without LangChain wrapping."""
    return _read_authoritative_guidelines_core(language, level)


def list_grounding_files(
    input_dir: Path | None = None,
    *,
    patterns: Sequence[str] = ("*.md", "*.txt", "*.markdown"),
) -> list[Path]:
    """Return sorted grounding files matching extensions under ``data/input``."""
    root = input_dir or _default_settings().data_input_dir
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(root.glob(pat)))
    # Unique preserve order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def read_grounding_file(path: Path, *, max_chars: int | None = None) -> str:
    """Read full text from a UTF-8 file (optional tail truncation for safety)."""
    text = path.read_text(encoding="utf-8")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n\n[…truncated…]"
    return text


def chunk_text(text: str, source_path: str, chunk_size: int = 2000, overlap: int = 200) -> list[GroundingChunk]:
    """
    Simple overlapping window chunker for RAG-style grounding.

    Replace with semantic chunking later if needed.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[GroundingChunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        cid = f"{source_path}#chunk-{idx}"
        chunks.append(GroundingChunk(chunk_id=cid, source_path=source_path, text=piece))
        if end >= n:
            break
        start = end - overlap
        idx += 1
    return chunks


def retrieve_for_query(
    query: str,
    *,
    input_dir: Path | None = None,
    top_k: int = 5,
) -> tuple[list[GroundingChunk], list[str]]:
    """
    Minimal deterministic retrieval: load all grounding files, chunk, score by keyword overlap.

    This is intentionally simple boilerplate; swap in embeddings + vector store for production.
    """
    _ = query  # reserved for embedding retrieval
    files = list_grounding_files(input_dir=input_dir)
    all_chunks: list[GroundingChunk] = []
    used_paths: list[str] = []
    for fp in files:
        used_paths.append(str(fp))
        body = read_grounding_file(fp)
        all_chunks.extend(chunk_text(body, str(fp)))

    # Trivial keyword overlap ranking
    q_tokens = set(query.lower().split())
    scored: list[tuple[float, GroundingChunk]] = []
    for ch in all_chunks:
        t_tokens = set(ch.text.lower().split())
        overlap = len(q_tokens & t_tokens)
        scored.append((float(overlap), ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:top_k]]
    return top, used_paths


def format_chunks_for_prompt(chunks: Sequence[GroundingChunk]) -> str:
    """Render chunks as a single prompt block with explicit source headers."""
    parts: list[str] = []
    for ch in chunks:
        parts.append(f"### SOURCE: {ch.chunk_id}\n{ch.text}\n")
    return "\n".join(parts)
