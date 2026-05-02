"""
Application configuration loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings."""

    google_api_key: str
    gemini_model: str
    data_input_dir: Path
    data_output_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro").strip()
        input_dir = Path(os.getenv("DATA_INPUT_DIR", str(_ROOT / "data" / "input")))
        output_dir = Path(os.getenv("DATA_OUTPUT_DIR", str(_ROOT / "data" / "output")))
        return cls(
            google_api_key=api_key,
            gemini_model=model,
            data_input_dir=input_dir.resolve(),
            data_output_dir=output_dir.resolve(),
        )


def get_settings() -> Settings:
    """Return cached-like settings (reload by calling again after env changes)."""
    return Settings.from_env()
