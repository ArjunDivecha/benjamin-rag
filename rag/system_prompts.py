"""System prompt loader utilities."""

from __future__ import annotations

from pathlib import Path


PROMPT_FILES = {
    "expert_network_brief": "expert_network_brief.md",
    "interview_guide": "interview_guide.md",
}


def load_system_prompt(objective: str) -> str:
    if objective not in PROMPT_FILES:
        raise FileNotFoundError(f"Unknown objective: {objective}")

    root = Path(__file__).resolve().parents[1]
    prompt_path = root / "system_prompts" / PROMPT_FILES[objective]
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise FileNotFoundError(f"System prompt file is empty: {prompt_path}")
    return content
