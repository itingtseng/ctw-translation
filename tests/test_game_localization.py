from __future__ import annotations

from pathlib import Path

import pandas as pd

from game_localization import (
    GameConfig,
    GameLocalizationAgent,
    approved_locks,
    approved_translation_memory,
    build_batch_context_questions,
    build_game_qa,
    build_game_records,
    build_review_table,
    build_translation_cards,
    default_character_bible,
    game_cache_key,
    game_context_risks,
    game_work_units,
    count_game_batches,
    infer_game_config,
    infer_string_id_clues,
    reviewed_export,
    select_review_rows,
    validate_character_bible,
)
from translation_agent import GlossaryEntry
from translation_agent import DemoTranslationBackend


class FakeGameBackend:
    def __init__(self):
        self.records = []
        self.style_calls = 0

    def translate_game(self, records, target_language, *, model=None):
        self.records.extend(records)
        return [
            f"{record['speaker']} says translated line {record['row_position']} {record['text']}"
            for record in records
        ]

    def evaluate_game_style(self, records, translations, target_language):
        self.style_calls += 1
        return [
            {"score": 91, "reason": "Voice matches.", "suggestion": translation}
            for translation in translations
        ]


def game_dataframe():
    return pd.DataFrame({
        "line_id": ["L001", "L002", "L003"],
        "scene_id": ["S1", "S1", "S1"],
        "speaker": ["Luna", "Alex", "Luna"],
        "listener": ["Alex", "Luna", "Alex"],
        "emotion": ["annoyed", "calm", "soft"],
        "source_text": ["好啊。", "好啊。", "下次別遲到了，{{player_name}}。"],
        "context": ["Late arrival", "Reply", "Reconciliation"],
        "character_limit": [50, 50, 80],
    })


def character_bible():
    bible = default_character_bible(game_dataframe(), GameConfig())
    bible.loc[bible["speaker"].eq("Luna"), "personality"] = "Proud but caring"
    bible.loc[bible["speaker"].eq("Alex"), "personality"] = "Calm and formal"
    return bible


def test_large_context_demo_has_complete_matching_character_bible():
    root = Path(__file__).resolve().parents[1]
    dialogue = pd.read_csv(root / "large_context_localization_demo.csv").fillna("")
    bible = pd.read_csv(root / "large_context_character_bible_demo.csv").fillna("")
    config = infer_game_config(dialogue)

    assert validate_character_bible(bible) == []
    assert set(dialogue[config.speaker].astype(str).str.strip()) == set(bible["speaker"])
    listeners = {
        value
        for value in dialogue[config.listener].astype(str).str.strip()
        if value
    }
    assert listeners <= set(bible["speaker"])

    # The large demo is mostly context-complete, with four deliberate gaps so
    # the Context review queue can be tested without faking low confidence.
    risks = game_context_risks(dialogue, config)
    assert sum(bool(risk) for risk in risks.values()) == 4


def test_qa_review_demo_exposes_issue_fix_and_final_translation_columns():
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "qa_review_demo.csv").fillna("")
    config = infer_game_config(source)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None), max_retries=0
    ).run(source, config, pd.DataFrame(), ["English"])
    review = build_review_table(source, result)

    assert set(result.qa_issues["type"]) == {
        "punctuation", "character_limit", "empty_translation"
    }
    punctuation = review[review["line_id"].eq("qa_punctuation_01")].iloc[0]
    assert punctuation["ai_translation"] == "Maintenance in progress"
    assert punctuation["qa_issue"].startswith("punctuation:")
    assert punctuation["suggested_fix"] == "Maintenance in progress."
    assert punctuation["reviewed_translation"] == "Maintenance in progress"

    length = review[review["line_id"].eq("qa_length_01")].iloc[0]
    assert length["qa_issue"].startswith("character_limit:")
    assert length["suggested_fix"] == "No safe automatic fix available"
    assert review[review["line_id"].eq("qa_ok_01")]["qa_issue"].iloc[0] == ""

    review.loc[review["line_id"].eq("qa_punctuation_01"), "reviewed_translation"] = (
        punctuation["suggested_fix"]
    )
    export = reviewed_export(source, result, review)
    assert "source__english__qa_issue" in export.columns
    assert "source__english__suggested_fix" in export.columns
    assert "source__english__final_translation" in export.columns
    assert export.loc[0, "source__english__final_translation"] == "Maintenance in progress."


def test_context_and_qa_review_demo_contains_both_review_queues():
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "context_qa_review_demo.csv").fillna("")
    config = infer_game_config(source)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None), max_retries=0
    ).run(source, config, pd.DataFrame(), ["English"])
    review = build_review_table(source, result)

    context_rows = set(review.loc[review["context_risk"].ne(""), "line_id"])
    qa_rows = set(review.loc[review["qa_issue"].ne(""), "line_id"])
    assert context_rows == {"context_only_01", "qa_punctuation_01"}
    assert qa_rows == {"qa_punctuation_01", "qa_length_01", "qa_empty_01"}
    assert context_rows & qa_rows == {"qa_punctuation_01"}


def test_review_signals_demo_contains_low_context_qa_and_failure():
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "review_signals_demo.csv").fillna("")
    config = infer_game_config(source)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0), max_retries=0
    ).run(source, config, pd.DataFrame(), ["English"])
    review = build_review_table(source, result)
    cards = build_translation_cards(source, result, review)

    assert set(cards.loc[cards["confidence_level"].eq("Low"), "line_id"]) == {
        "context_low_01"
    }
    assert set(review.loc[review["context_risk"].ne(""), "line_id"]) == {
        "context_low_01"
    }
    assert set(result.qa_issues["line_id"]) == {"qa_punctuation_01"}
    assert set(result.failures["line_id"]) == {"failed_value_01"}
    assert cards.loc[
        cards["line_id"].eq("failed_value_01"), "confidence_level"
    ].iloc[0] == "Failed"


def test_character_bible_upload_rejects_unrelated_or_ambiguous_data():
    dialogue_file = pd.DataFrame({"speaker": ["Luna"], "source_text": ["你好"]})
    assert "Add at least one character guidance column" in validate_character_bible(dialogue_file)[0]
    assert validate_character_bible(pd.DataFrame({"name": ["Luna"], "personality": ["Proud"]})) == [
        "Missing required column: speaker."
    ]
    duplicate_bible = pd.DataFrame({
        "speaker": ["Luna", "Luna"],
        "personality": ["Proud", "Caring"],
    })
    assert any("Duplicate speaker(s): Luna" in error for error in validate_character_bible(duplicate_bible))
    assert validate_character_bible(character_bible()) == []


def test_context_and_cache_key_include_character_scene_and_neighboring_lines():
    records = build_game_records(game_dataframe(), GameConfig(), character_bible())
    assert records[0]["next_lines"][0]["speaker"] == "Alex"
    assert records[1]["previous_lines"][0]["speaker"] == "Luna"
    assert records[0]["character"]["personality"] == "Proud but caring"
    assert game_cache_key(records[0], "English") != game_cache_key(records[1], "English")


def test_string_package_context_risks_are_deterministic_and_notes_resolve_them():
    source = pd.DataFrame({
        "line_id": ["A", "B", "C"],
        "scene_id": ["chest", "inventory", ""],
        "speaker": ["Player", "System", "Player"],
        "source_text": ["打開", "打開", "它不見了。"],
    })
    config = GameConfig(listener="", emotion="", context="")
    risks = game_context_risks(source, config)
    assert "Short string may have multiple meanings" in risks[0]
    assert "Same source appears in different contexts" not in risks[0]
    assert "Referent may be unclear" in risks[2]
    assert game_context_risks(source, config, {2: "Quest item; it refers to the moonstone."})[2] == ""


def test_scene_and_speaker_work_units_control_batch_count():
    units = game_work_units(game_dataframe(), GameConfig())
    assert list(units[["Scene", "Speaker", "Lines"]].itertuples(index=False, name=None)) == [
        ("S1", "Luna", 2),
        ("S1", "Alex", 1),
    ]
    assert count_game_batches(game_dataframe(), GameConfig(), batch_size=20) == 2


def test_game_schema_mapping_accepts_common_column_names():
    config = infer_game_config(pd.DataFrame({"Character": ["Luna"], "Dialogue": ["你好"]}))
    assert config.speaker == "Character"
    assert config.source_text == "Dialogue"


def test_string_package_schema_accepts_key_source_without_speaker():
    source = pd.DataFrame({
        "key": ["btn_open_chest"],
        "source": ["Open"],
        "zh-TW": [""],
        "context": ["Chest interaction button"],
        "char_limit": [4],
        "screen": ["treasure_room"],
    })
    config = infer_game_config(source)
    assert config.line_id == "key"
    assert config.source_text == "source"
    assert config.speaker == ""
    assert config.scene_id == "screen"
    assert config.character_limit == "char_limit"
    records = build_game_records(source, config, pd.DataFrame())
    assert records[0]["speaker"] == ""
    assert records[0]["id_clues"] == "button · open · chest"
    assert infer_string_id_clues("dlg_mira_03") == "dialogue · mira"


def test_translation_cards_expose_confidence_provenance_tm_and_question_batch():
    source = pd.DataFrame({
        "key": ["btn_open_chest", "ui_inventory_open", "dlg_mira_04"],
        "screen": ["treasure", "inventory", "chapter_03"],
        "speaker": ["System", "System", "Mira"],
        "source": ["Open", "Open", "Take it."],
        "context": ["Chest interaction", "", ""],
        "char_limit": [4, 4, 12],
    })
    config = infer_game_config(source)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None), max_retries=0
    ).run(source, config, pd.DataFrame(), ["Japanese"])
    review = build_review_table(source, result)
    assert "rerun_selected" in review.columns
    review.loc[0, "status"] = "Approved"
    review.loc[0, "reviewed_translation"] = "開く"
    assert approved_translation_memory(review)[("Open", "Japanese")] == "開く"

    cards = build_translation_cards(source, result, review)
    inventory = cards[cards["line_id"].eq("ui_inventory_open")].iloc[0]
    assert inventory["confidence_level"] == "Low"
    assert "ID inference (unverified)" in inventory["context_sources"]
    assert "Location: inventory" in inventory["context_sources"]
    assert "Related strings: None found" in inventory["context_sources"]
    assert "String grouping:" not in inventory["context_sources"]
    assert "different contexts" not in inventory["ambiguity"]
    assert "Short string may have multiple meanings" in inventory["ambiguity"]
    assert inventory["tm_match"] == "開く"
    review.loc[review["line_id"].eq("dlg_mira_04"), "status"] = "Needs context"
    review.loc[review["line_id"].eq("dlg_mira_04"), "reviewer_comment"] = (
        "Please confirm what 'it' refers to."
    )
    cards = build_translation_cards(source, result, review)
    questions = build_batch_context_questions(cards)
    assert not questions.empty
    assert "ui_inventory_open" in " ".join(questions["affected_string_ids"])
    assert "dlg_mira_04" in " ".join(questions["affected_string_ids"])
    assert "Please confirm what 'it' refers to." in " ".join(questions["question"])
    manual_only = cards[cards["line_id"].eq("dlg_mira_04")].copy()
    manual_only.loc[:, "ambiguity"] = ""
    manual_only.loc[:, "reviewer_comment"] = ""
    manual_questions = build_batch_context_questions(manual_only)
    assert len(manual_questions) == 1
    assert manual_questions.iloc[0]["affected_string_ids"] == "dlg_mira_04"


def test_game_translation_uses_character_context_and_optional_style_evaluation():
    backend = FakeGameBackend()
    result = GameLocalizationAgent(backend, batch_size=2, max_retries=0).run(
        game_dataframe(),
        GameConfig(),
        character_bible(),
        ["English"],
        evaluate_style=True,
        context_overrides={0: "Luna is confronting the player at the temple entrance."},
    )
    assert result.game_config["speaker"] == "speaker"
    assert result.dataframe["source_text__english"].iloc[0].startswith("Luna says translated line 0")
    assert result.dataframe["source_text__english"].iloc[1].startswith("Alex says translated line 1")
    assert backend.records[0]["emotion"] == "annoyed"
    assert backend.records[0]["context_note"].startswith("Luna is confronting")
    assert len(result.style_evaluations) == 3
    assert result.style_evaluations["score"].eq(91).all()


def test_demo_failure_is_isolated_and_targeted_retry_succeeds():
    source = game_dataframe().copy()
    source.loc[1, "source_text"] = "[FAIL] 這一行第一次應該失敗。"
    bible = character_bible()
    first = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker="[FAIL]"),
        batch_size=20,
        max_retries=0,
    ).run(source, GameConfig(), bible, ["English"])
    assert len(first.failures) == 1
    assert first.metrics.coverage == 2 / 3
    failed_position = int(first.failures.iloc[0]["row_position"])

    retried = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None),
        batch_size=20,
        max_retries=0,
    ).run(
        source,
        GameConfig(),
        bible,
        ["English"],
        row_positions=[failed_position],
        base_result=first,
    )
    assert retried.failures.empty
    assert retried.dataframe.loc[failed_position, "source_text__english"] == "[FAIL] 這一行第一次應該失敗。"


def test_game_qa_detects_glossary_placeholder_tag_limit_and_untranslated_text():
    source = pd.DataFrame({
        "line_id": ["1"],
        "scene_id": ["S"],
        "speaker": ["Luna"],
        "source_text": ["<b>星核</b> {{name}}"],
        "character_limit": [5],
    })
    translated = source.assign(source_text__english=["星核"])
    issues = build_game_qa(
        source,
        translated,
        GameConfig(),
        ["English"],
        [GlossaryEntry("星核", "English", "Astral Core", "Item", "Official")],
    )
    assert {
        "untranslated_chinese",
        "placeholder_mismatch",
        "rich_text_mismatch",
        "glossary_violation",
    }.issubset(set(issues["type"]))


def test_review_workbench_locks_approved_lines_and_exports_reviewed_values():
    backend = FakeGameBackend()
    source = game_dataframe()
    result = GameLocalizationAgent(backend, batch_size=5, max_retries=0).run(
        source, GameConfig(), character_bible(), ["English"]
    )
    review = build_review_table(source, result)
    review.loc[review["row_position"].eq(0), "status"] = "Approved"
    review.loc[review["row_position"].eq(0), "reviewed_translation"] = "Fine."
    review.loc[review["row_position"].eq(1), "status"] = "Needs revision"
    review.loc[review["row_position"].eq(1), "selected"] = True

    assert approved_locks(review) == {(0, "English")}
    assert select_review_rows(review, "Selected lines") == [1]
    assert select_review_rows(review, "Character", "Luna") == [0, 2]
    assert select_review_rows(review, "Needs revision") == [1]

    review.loc[review["row_position"].eq(2), "status"] = "Keep source text"
    assert approved_locks(review) == {(0, "English"), (2, "English")}

    retry_backend = FakeGameBackend()
    rerun = GameLocalizationAgent(retry_backend, batch_size=5, max_retries=0).run(
        source,
        GameConfig(),
        character_bible(),
        ["English"],
        row_positions=[0, 1],
        base_result=result,
        approved_locks=approved_locks(review),
    )
    assert [record["row_position"] for record in retry_backend.records] == [1]
    merged_review = build_review_table(source, rerun, review)
    assert merged_review.loc[merged_review["row_position"].eq(0), "reviewed_translation"].iloc[0] == "Fine."
    export = reviewed_export(source, rerun, merged_review)
    assert export.loc[0, "source_text__english__reviewed"] == "Fine."
    assert export.loc[0, "source_text__english__status"] == "Approved"
    assert "source_text__english__notes" in export.columns
    assert "source_text__english__needs_context" in export.columns
    assert "source_text__english__keep_source_text" in export.columns
    assert "source_text__english__confidence_percent" in export.columns
    assert bool(export.loc[2, "source_text__english__keep_source_text"])
    assert 0 < export.loc[0, "source_text__english__confidence_percent"] <= 100


def test_context_risk_rows_are_proactively_escalated_to_a_stronger_model():
    class RecordingBackend:
        def __init__(self):
            self.calls: list[tuple[list[dict], str | None]] = []

        def translate_game(self, records, target_language, *, model=None):
            self.calls.append((list(records), model))
            return [f"{target_language}: {record['text']}" for record in records]

    source = pd.DataFrame({
        "line_id": ["L1", "L2"],
        "speaker": ["Luna", "Luna"],
        "scene_id": ["", "S1"],
        "context": ["", "Calm afternoon chat"],
        "source_text": ["這個。", "今天天氣真好，適合出去玩。"],
    })
    backend = RecordingBackend()
    agent = GameLocalizationAgent(backend, batch_size=5, max_retries=0, escalation_model="gpt-4o")
    result = agent.run(source, GameConfig(), pd.DataFrame(), ["English"])
    model_by_text = {call[0][0]["text"]: call[1] for call in backend.calls}
    assert model_by_text["這個。"] == "gpt-4o"
    assert model_by_text["今天天氣真好，適合出去玩。"] is None
    assert result.metrics.escalated_batches == 1


def test_escalated_batches_counts_once_even_when_proactive_and_retry_both_apply():
    class FlakyBackend:
        def __init__(self):
            self.calls: list[str | None] = []

        def translate_game(self, records, target_language, *, model=None):
            self.calls.append(model)
            if len(self.calls) == 1:
                raise RuntimeError("transient failure")
            return [f"{target_language}: {record['text']}" for record in records]

    source = pd.DataFrame({
        "line_id": ["L1"],
        "speaker": ["Luna"],
        "scene_id": [""],
        "context": [""],
        "source_text": ["這個。"],
    })
    backend = FlakyBackend()
    result = GameLocalizationAgent(backend, batch_size=5, max_retries=1, escalation_model="gpt-4o").run(
        source, GameConfig(), pd.DataFrame(), ["English"]
    )
    assert backend.calls == ["gpt-4o", "gpt-4o"]
    assert result.metrics.escalated_batches == 1


def test_back_translation_check_flags_low_similarity_round_trip():
    class RoundTripBackend:
        def translate_game(self, records, target_language, *, model=None):
            texts = [record["text"] for record in records]
            if target_language == "English":
                return [f"EN:{text}" for text in texts]
            return [
                "完全不同的意思" if text == "EN:壞掉的翻譯" else text.removeprefix("EN:")
                for text in texts
            ]

    source = pd.DataFrame({
        "line_id": ["L1", "L2"],
        "speaker": ["Luna", "Luna"],
        "scene_id": ["S1", "S1"],
        "context": ["chat", "chat"],
        "source_text": ["正常翻譯", "壞掉的翻譯"],
    })
    result = GameLocalizationAgent(RoundTripBackend(), batch_size=10, max_retries=0).run(
        source,
        GameConfig(),
        pd.DataFrame(),
        ["English"],
        evaluate_hallucination=True,
        hallucination_sample_rate=1.0,
        hallucination_similarity_threshold=0.6,
    )
    flagged = result.qa_issues[result.qa_issues["type"] == "possible_hallucination"]
    assert list(flagged["line_id"]) == ["L2"]


def test_hallucination_check_is_skipped_by_default_and_in_demo_mode():
    source = pd.DataFrame({
        "line_id": ["L1"],
        "speaker": ["Luna"],
        "scene_id": ["S1"],
        "context": ["chat"],
        "source_text": ["正常翻譯"],
    })
    result = GameLocalizationAgent(FakeGameBackend(), batch_size=10, max_retries=0).run(
        source, GameConfig(), pd.DataFrame(), ["English"]
    )
    assert "possible_hallucination" not in set(result.qa_issues.get("type", []))

    demo_result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None), batch_size=10, max_retries=0
    ).run(
        source, GameConfig(), pd.DataFrame(), ["English"],
        evaluate_hallucination=True, hallucination_sample_rate=1.0,
    )
    assert "possible_hallucination" not in set(demo_result.qa_issues.get("type", []))
