"""Standalone evaluation harness for `evaluation/eval_set.csv`.

This scores *raw* candidate-model output against a small hand-labeled gold set on four
axes: exact-match rate, character-level similarity, placeholder preservation, and
glossary compliance. It intentionally does not run the full `TranslationAgent` pipeline:
this project's `protect_tokens`/`glossary_replacements` layer (see translation_agent.py)
already makes placeholder loss and glossary drift structurally impossible once a glossary
is attached, so exercising that path here would only prove the harness against itself.
This script instead evaluates a candidate backend/model/prompt directly and independently,
the way you would before trusting it enough to sit behind that protection layer, or when
evaluating a provider that has no such layer.

Run directly for a printed report: `python evaluate.py`
Import `evaluate_language` from tests for a regression-style assertion.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

EVAL_SET_PATH = Path(__file__).parent / "evaluation" / "eval_set.csv"

PLACEHOLDER_TOKENS = ("{{count}}", "{{player_name}}")

# Simulates raw candidate-model output for English, deliberately including the kind of
# mistakes a bare LLM call (without this project's deterministic protection layer) can
# make: a glossary slip on E04, and a dropped placeholder on E05. Everything else is a
# faithful, if not word-for-word, rendering of the gold translation.
CANDIDATE_ENGLISH = {
    "E01": "Welcome back, hero.",
    "E02": "What would you like to do today?",
    "E03": "The Astral Core is ready.",
    "E04": "Please go to the Moon Temple to complete the quest.",
    "E05": "You have some items left.",
    "E06": "The merchant is not here today.",
    "E07": "Watch out! A monster has appeared!",
    "E08": "This quest is somewhat difficult.",
    "E09": "Welcome to the Eldora Shop.",
    "E10": "Your level has increased!",
    "E11": "The Astral Core's energy is weakening.",
    "E12": "Please enter your name: {{player_name}}",
    "E13": "Battle start!",
    "E14": "Are you sure you want to leave?",
    "E15": "Thank you for your help, hero.",
    "E16": "The gate of the Temple of Luna has opened.",
}


@dataclass(frozen=True)
class EvalRow:
    id: str
    source: str
    glossary_term: str
    glossary_translation: str
    gold_english: str
    gold_japanese: str


@dataclass(frozen=True)
class EvalReport:
    rows: int
    exact_match_rate: float
    avg_similarity: float
    placeholder_preservation_rate: float
    glossary_compliance_rate: float
    mismatches: list[str]


def load_eval_set() -> list[EvalRow]:
    with EVAL_SET_PATH.open(encoding="utf-8-sig") as handle:
        return [
            EvalRow(
                id=row["id"],
                source=row["source"],
                glossary_term=row["glossary_term"],
                glossary_translation=row["glossary_translation"],
                gold_english=row["gold_english"],
                gold_japanese=row["gold_japanese"],
            )
            for row in csv.DictReader(handle)
        ]


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def evaluate_language(rows: list[EvalRow], predictions: dict[str, str], gold_field: str) -> EvalReport:
    mismatches: list[str] = []
    exact_matches = 0
    similarities: list[float] = []
    placeholder_checks: list[bool] = []
    glossary_checks: list[bool] = []

    for row in rows:
        gold = getattr(row, gold_field)
        predicted = predictions.get(row.id, "")
        similarity = text_similarity(predicted, gold)
        similarities.append(similarity)
        is_exact = predicted.strip() == gold.strip()
        exact_matches += int(is_exact)
        if not is_exact:
            mismatches.append(f"{row.id}: predicted={predicted!r} gold={gold!r} (similarity={similarity:.0%})")

        expected_placeholders = [token for token in PLACEHOLDER_TOKENS if token in row.source]
        if expected_placeholders:
            placeholder_checks.append(all(token in predicted for token in expected_placeholders))

        if row.glossary_term:
            glossary_checks.append(row.glossary_translation in predicted)

    return EvalReport(
        rows=len(rows),
        exact_match_rate=exact_matches / len(rows) if rows else 1.0,
        avg_similarity=sum(similarities) / len(similarities) if similarities else 1.0,
        placeholder_preservation_rate=(
            sum(placeholder_checks) / len(placeholder_checks) if placeholder_checks else 1.0
        ),
        glossary_compliance_rate=sum(glossary_checks) / len(glossary_checks) if glossary_checks else 1.0,
        mismatches=mismatches,
    )


def main() -> None:
    rows = load_eval_set()
    report = evaluate_language(rows, CANDIDATE_ENGLISH, "gold_english")
    print(f"Evaluated {report.rows} rows (English)")
    print(f"  exact_match_rate           = {report.exact_match_rate:.0%}")
    print(f"  avg_similarity             = {report.avg_similarity:.0%}")
    print(f"  placeholder_preservation   = {report.placeholder_preservation_rate:.0%}")
    print(f"  glossary_compliance        = {report.glossary_compliance_rate:.0%}")
    if report.mismatches:
        print("\nMismatches vs gold:")
        for line in report.mismatches:
            print(f"  - {line}")


if __name__ == "__main__":
    sys.exit(main() or 0)
