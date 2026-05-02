"""
Entry point: load configuration, run the curriculum graph, persist JSON to ``data/output``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .graph import compile_curriculum_graph
from .schemas import GrammarCurriculum, VocabularyCurriculum
from .state import initial_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous Curriculum Generator (LangGraph + Gemini)")
    p.add_argument("--language", default="English", help="Target language name or BCP-47 tag")
    p.add_argument("--level", default="B1", help="CEFR level (A1–C2)")
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output path prefix: writes <stem>_vocabulary.json and <stem>_grammar.json "
            "next to a .json path, or two files under a directory. Default: data/output/curriculum_<ts>_*.json"
        ),
    )
    return p.parse_args()


def resolve_output_paths(out: str | None, ts: str, default_dir: Path) -> tuple[Path, Path]:
    """Return (vocabulary_json, grammar_json) paths."""
    if not out:
        return (
            default_dir / f"curriculum_{ts}_vocabulary.json",
            default_dir / f"curriculum_{ts}_grammar.json",
        )
    p = Path(out)
    if p.suffix.lower() == ".json":
        stem = p.stem
        parent = p.parent
        return (
            parent / f"{stem}_vocabulary.json",
            parent / f"{stem}_grammar.json",
        )
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"curriculum_{ts}_vocabulary.json",
        p / f"curriculum_{ts}_grammar.json",
    )


def save_bundle(
    vocabulary: VocabularyCurriculum | None,
    grammar: GrammarCurriculum | None,
    vocab_path: Path,
    grammar_path: Path,
    errors: list[str],
) -> None:
    """Serialize validated models to two JSON files, or write error envelope to both paths."""
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    grammar_path.parent.mkdir(parents=True, exist_ok=True)

    err_payload = {
        "error": "No valid curriculum produced",
        "validation_errors": errors,
    }
    if vocabulary is not None and grammar is not None:
        vocab_path.write_text(
            json.dumps(vocabulary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        grammar_path.write_text(
            json.dumps(grammar.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        text = json.dumps(err_payload, ensure_ascii=False, indent=2)
        vocab_path.write_text(text, encoding="utf-8")
        grammar_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    settings.data_output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    vocab_out, grammar_out = resolve_output_paths(args.out, ts, settings.data_output_dir)

    graph = compile_curriculum_graph()
    seed = initial_state(language=args.language, cefr_level=args.level)
    result = graph.invoke(seed)

    vocabulary = result.get("vocabulary_curriculum")
    grammar = result.get("grammar_curriculum")
    errs = list(result.get("validation_errors") or [])

    if isinstance(vocabulary, VocabularyCurriculum) and isinstance(grammar, GrammarCurriculum):
        save_bundle(vocabulary, grammar, vocab_out, grammar_out, [])
    else:
        save_bundle(None, None, vocab_out, grammar_out, errs)

    print(f"Wrote: {vocab_out}")
    print(f"Wrote: {grammar_out}")


if __name__ == "__main__":
    main()
