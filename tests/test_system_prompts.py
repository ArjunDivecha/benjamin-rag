from pathlib import Path

import pytest

from rag.system_prompts import load_system_prompt


def test_prompt_files_exist_and_non_empty():
    root = Path(__file__).resolve().parents[1]
    p1 = root / "system_prompts" / "expert_network_brief.md"
    p2 = root / "system_prompts" / "interview_guide.md"
    p3 = root / "system_prompts" / "insights_qa.md"
    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert p1.read_text(encoding="utf-8").strip()
    assert p2.read_text(encoding="utf-8").strip()
    assert p3.read_text(encoding="utf-8").strip()


def test_load_expert_network_prompt_contains_screening():
    text = load_system_prompt("expert_network_brief")
    assert "screening" in text.lower()


def test_load_interview_guide_prompt_contains_stakeholder():
    text = load_system_prompt("interview_guide")
    assert "stakeholder" in text.lower()


def test_load_insights_prompt_contains_evidence():
    text = load_system_prompt("insights_qa")
    assert "evidence" in text.lower()


def test_load_nonexistent_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_system_prompt("nonexistent")
