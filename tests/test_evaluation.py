from __future__ import annotations

from evaluate import CANDIDATE_ENGLISH, evaluate_language, load_eval_set


def test_eval_set_loads_and_has_glossary_and_placeholder_coverage():
    rows = load_eval_set()
    assert len(rows) >= 15
    assert any(row.glossary_term for row in rows)
    assert any("{{" in row.source for row in rows)


def test_evaluation_harness_catches_known_glossary_and_placeholder_regressions():
    rows = load_eval_set()
    report = evaluate_language(rows, CANDIDATE_ENGLISH, "gold_english")

    # The metric is meaningful only if it can fail: assert it actually distinguishes
    # good rows from the two seeded regressions rather than trivially reporting 0% or 100%.
    assert 0.0 < report.exact_match_rate < 1.0
    assert 0.0 < report.placeholder_preservation_rate < 1.0
    assert 0.0 < report.glossary_compliance_rate < 1.0
    assert report.avg_similarity > 0.9

    assert any(line.startswith("E04:") for line in report.mismatches)
    assert any(line.startswith("E05:") for line in report.mismatches)


def test_evaluation_harness_reports_perfect_scores_for_matching_predictions():
    rows = load_eval_set()
    perfect = {row.id: row.gold_english for row in rows}
    report = evaluate_language(rows, perfect, "gold_english")
    assert report.exact_match_rate == 1.0
    assert report.avg_similarity == 1.0
    assert report.placeholder_preservation_rate == 1.0
    assert report.glossary_compliance_rate == 1.0
    assert report.mismatches == []
