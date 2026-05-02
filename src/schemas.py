"""
Strict Pydantic models for vocabulary, grammar, and export bundles.

Design goals:
- JSON-serializable, schema-stable outputs suitable for downstream validation.
- CEFR level as a constrained literal set.
- Separate vocabulary vs grammar root documents; uniqueness enforced at validation time.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class CEFRLevel(str, Enum):
    """Common European Framework of Reference for Languages levels."""

    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class PartOfSpeech(str, Enum):
    """Controlled vocabulary for POS tags (extend as needed)."""

    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    ARTICLE = "article"
    NUMERAL = "numeral"
    OTHER = "other"


class VocabularyItem(BaseModel):
    """A single lexical unit with CEFR alignment and optional grounding references."""

    lemma: str = Field(..., min_length=1, description="Canonical dictionary form.")
    pos: PartOfSpeech = Field(..., description="Part of speech.")
    translation_en: str | None = Field(
        default=None,
        description="English gloss for L2 learners (optional).",
    )
    cefr_level: CEFRLevel = Field(..., description="Target CEFR band for this item.")
    topic_tags: list[str] = Field(default_factory=list, description="Thematic tags.")
    source_refs: list[str] = Field(
        default_factory=list,
        description="Identifiers of grounding chunks (e.g. filenames or chunk IDs).",
    )
    notes: str | None = Field(default=None, description="Pedagogical or usage notes.")


class VocabularySection(BaseModel):
    """Grouped vocabulary for a curriculum unit or theme."""

    title: str = Field(..., min_length=1)
    description: str | None = None
    items: list[VocabularyItem] = Field(default_factory=list)


class GrammarExample(BaseModel):
    """Illustrative sentence or fragment."""

    text: str = Field(..., min_length=1)
    gloss_en: str | None = None
    source_refs: list[str] = Field(default_factory=list)


class GrammarRule(BaseModel):
    """A rule or pattern with CEFR alignment."""

    rule_id: str = Field(..., min_length=1, description="Stable ID within the curriculum.")
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    cefr_level: CEFRLevel
    examples: list[GrammarExample] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    related_vocabulary_lemmas: list[str] = Field(
        default_factory=list,
        description="Cross-links to vocabulary lemmas where helpful.",
    )


class GrammarSection(BaseModel):
    """Container for grammar rules under a heading."""

    title: str = Field(..., min_length=1)
    rules: list[GrammarRule] = Field(default_factory=list)


class CurriculumMetadata(BaseModel):
    """Provenance and pipeline metadata (non-LLM-truth claims)."""

    generated_at_iso: str = Field(..., description="UTC ISO-8601 timestamp.")
    pipeline_version: str = Field(default="0.1.0")
    language: str = Field(..., min_length=2, description="Target language name or BCP-47 tag.")
    cefr_target: CEFRLevel
    grounding_sources: list[str] = Field(
        default_factory=list,
        description="Input files or corpora used for RAG/grounding.",
    )


class VocabularyCurriculum(BaseModel):
    """
    Root document for vocabulary-only JSON export.

    Lemmas are de-duplicated using Unicode case-folding and leading/trailing whitespace
    normalization so surface variants do not slip through.
    """

    metadata: CurriculumMetadata
    vocabulary_sections: list[VocabularySection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_lemmas(self) -> VocabularyCurriculum:
        seen: dict[str, str] = {}
        for section in self.vocabulary_sections:
            for item in section.items:
                key = item.lemma.strip().casefold()
                if key in seen:
                    first = seen[key]
                    raise ValueError(
                        f"Duplicate vocabulary lemma: {item.lemma!r} "
                        f"(same normalized form as {first!r}); lemmas must be unique across the curriculum."
                    )
                seen[key] = item.lemma
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "metadata": {
                        "generated_at_iso": "2026-05-02T12:00:00Z",
                        "pipeline_version": "0.1.0",
                        "language": "Turkish",
                        "cefr_target": "B1",
                        "grounding_sources": ["cefr_reference.md"],
                    },
                    "vocabulary_sections": [],
                }
            ]
        }
    }


class GrammarCurriculum(BaseModel):
    """
    Root document for grammar-only JSON export.

    ``rule_id`` values must be unique across all sections (whitespace-stripped equality).
    """

    metadata: CurriculumMetadata
    grammar_sections: list[GrammarSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> GrammarCurriculum:
        seen: set[str] = set()
        for section in self.grammar_sections:
            for rule in section.rules:
                rid = rule.rule_id.strip()
                if rid in seen:
                    raise ValueError(
                        f"Duplicate grammar rule_id: {rule.rule_id!r}; "
                        f"rule identifiers must be unique across the curriculum."
                    )
                seen.add(rid)
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "metadata": {
                        "generated_at_iso": "2026-05-02T12:00:00Z",
                        "pipeline_version": "0.1.0",
                        "language": "Turkish",
                        "cefr_target": "B1",
                        "grounding_sources": ["cefr_reference.md"],
                    },
                    "grammar_sections": [],
                }
            ]
        }
    }
