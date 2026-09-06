from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from translation_agent import TranslationMetrics, TranslationResult, profile_columns
from translation_agent import DemoTranslationBackend
from game_localization import (
    GameConfig,
    GameLocalizationAgent,
    build_review_table,
    default_character_bible,
)
from app import REVIEW_REPORT_COLUMNS as REVIEW_REPORT_COLUMNS_FOR_TEST
from app import format_length_field


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def make_document(document_id: str, name: str, dataframe: pd.DataFrame) -> dict:
    return {
        "id": document_id,
        "name": name,
        "project_name": Path(name).stem,
        "size": 100,
        "dataframe": dataframe,
        "profiles": profile_columns(dataframe),
        "messages": [{"role": "assistant", "content": f"{name} conversation"}],
        "result": None,
        "unread": False,
        "job_id": None,
        "draft": "",
    }


def test_empty_app_prompts_for_a_file():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    assert not app.exception
    assert any(subheader.value == "Workspace" for subheader in app.subheader)
    assert any(
        caption.value
        == "Your file summary, translation progress, preview, and results will appear here."
        for caption in app.caption
    )
    assert not app.get("file_uploader")
    assert any(button.key == "open_onboarding_translation_upload" for button in app.button)
    assert any(button.key == "open_onboarding_bible_upload" for button in app.button)
    assert next(
        button for button in app.button if button.key == "new_dialogue_project"
    ).proto.type == "secondary"
    assert any(
        markdown.value.startswith("Hi! I’m your localization agent.")
        for markdown in app.markdown
    )
    assert not any(button.key == "open_pending_glossary" for button in app.button)
    assert any(chat.key == "onboarding_chat_input" for chat in app.chat_input)
    onboarding_languages = next(
        button for button in app.button if button.key == "open_onboarding_languages"
    )
    assert onboarding_languages.proto.type == "primary"
    onboarding_languages.click().run(timeout=15)
    assert any(
        field.key == "onboarding_conversation_languages"
        for field in app.get("button_group")
    )

    assert not any(button.key == "onboarding_open_translation" for button in app.button)
    assert not any(button.key == "onboarding_open_bible" for button in app.button)


def test_review_grid_offers_review_signals_and_sort_options_without_review_mode():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'value="QA issues first">Sort by: QA issues first' in source
    assert 'value="Low confidence first">Sort by: Low confidence first' in source
    assert 'aria-label="Review mode"' not in source
    assert 'Review mode:' not in source
    assert '<th class="confidence-column">Confidence</th>' in source
    assert '<th class="confidence-column">Confidence score</th>' not in source
    assert '<th class="confidence-column">Confidence level</th>' not in source
    assert 'Yellow: context review' in source
    assert 'Red: QA issue' in source
    assert 'Blue: low confidence' in source
    assert '<th class="id-column">String ID</th>' in source
    assert '<th class="language-column">Language</th>' in source
    assert '<th class="speaker-column">Speaker</th>' in source
    assert '<th class="text-column">Developer note</th>' in source
    assert '<th class="length-column">Length</th>' in source
    assert '<th class="context-column">Context review</th>' in source
    assert '<th class="qa-column">QA issue</th>' in source
    assert '<th class="text-column">Suggested fix</th>' in source
    assert '<th class="failure-column">Failure reason</th>' in source
    assert '<th class="text-column">Final translation</th>' in source
    assert '<th class="keep-column">Keep source text</th>' not in source
    assert 'keep.textContent = "Keep source"' in source
    assert source.index('originalLayout.appendChild(originalText)') < source.index(
        'originalLayout.appendChild(keep)'
    )
    shared_headers = [
        '<th class="id-column">String ID</th>',
        '<th class="language-column">Language</th>',
        '<th class="speaker-column">Speaker</th>',
        '<th class="text-column">Original</th>',
        '<th class="text-column">Developer note</th>',
        '<th class="text-column">AI translation</th>',
        '<th class="text-column">Final translation</th>',
        '<th class="context-column">Context review</th>',
        '<th class="flag-column">Needs context</th>',
        '<th class="qa-column">QA issue</th>',
        '<th class="text-column">Suggested fix</th>',
        '<th class="length-column">Length</th>',
        '<th class="confidence-column">Confidence</th>',
        '<th class="failure-column">Failure reason</th>',
    ]
    assert [source.index(header) for header in shared_headers] == sorted(
        source.index(header) for header in shared_headers
    )
    details_source = source.split("const showDetails = (row) => {", 1)[1].split(
        "const addTextCell", 1
    )[0]
    assert '["Confidence", row.confidence]' not in details_source
    assert '["Developer note", row.developer_note]' not in details_source
    for detail_label in (
        "Scene", "Speaker → Listener", "Emotion", "Previous dialogue",
        "Next dialogue", "Glossary hits", "TM matches", "Placeholder details",
        "Translation run", "Review audit", "Screenshot",
    ):
        assert detail_label in details_source


def test_length_field_always_shows_current_count_and_limit_slot():
    assert format_length_field("翻譯", "12") == "2 / 12"
    assert format_length_field("Translation", "") == "11 / —"


def test_context_review_report_precedes_qa_report_and_is_downloadable():
    source = pd.read_csv(
        APP_PATH.parent / "samples" / "feature-tests" / "review_signals_demo.csv"
    ).fillna("")
    config = GameConfig(
        line_id="key", scene_id="screen", speaker="speaker", listener="listener",
        emotion="emotion", source_text="source", context="context",
        character_limit="char_limit",
    )
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0), max_retries=0
    ).run(source, config, pd.DataFrame(), ["English"])
    document = make_document("game", "review_signals_demo.csv", source)
    document.update({
        "game_mode": True,
        "game_config": result.game_config,
        "character_bible": pd.DataFrame(),
        "result": result,
        "target_languages": ["English"],
        "result_target_languages": ["English"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": document}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)

    labels = [expander.label for expander in app.expander]
    assert "Context review (1)" in labels
    assert "Game QA issues (1)" in labels
    assert "Failure triage (1 unresolved)" in labels
    assert labels.index("Failure triage (1 unresolved)") < labels.index(
        "Context review (1)"
    )
    assert labels.index("Context review (1)") < labels.index("Game QA issues (1)")
    report_tables = [
        table.value for table in app.dataframe
        if list(table.value.columns) == REVIEW_REPORT_COLUMNS_FOR_TEST
    ]
    assert len(report_tables) == 2
    failure_editor = next(
        editor.value for editor in app.dataframe
        if (editor.key or "").startswith("failure_triage_game_")
    )
    assert list(failure_editor.columns[:len(REVIEW_REPORT_COLUMNS_FOR_TEST)]) == (
        REVIEW_REPORT_COLUMNS_FOR_TEST
    )
    assert any(
        button.label == "Download context review CSV"
        for button in app.get("download_button")
    )


def test_failure_triage_and_execution_history_render_before_review_launcher():
    source = APP_PATH.read_text(encoding="utf-8")
    function = source.split("def render_post_setup_result", 1)[1].split(
        "\ndef open_new_project_upload", 1
    )[0]
    assert function.index("render_failure_triage") < function.index(
        "render_execution_log"
    ) < function.index("render_game_review_launcher")


def test_active_project_uses_native_chat_input():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {
        "one": make_document("one", "first.csv", dataframe),
    }
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    assert not app.exception
    assert not any(
        markdown.value.startswith("**Terminology rules**") for markdown in app.markdown
    )
    assert not any(button.key == "open_glossary_one" for button in app.button)
    assert any(chat.key == "chat_input_one" for chat in app.chat_input)
    assert next(
        button for button in app.button
        if button.key == "open_translation_upload_one"
    ).proto.type == "secondary"
    assert next(
        button for button in app.button
        if button.key == "open_translation_upload_one"
    ).label == "Update translation data"
    assert next(
        button for button in app.button
        if button.key == "open_conversation_languages_one"
    ).proto.type == "primary"
    assert not any(button.key == "open_bible_upload_one" for button in app.button)
    assert not any(button.key == "translate_one" for button in app.button)


def test_general_csv_uses_current_conversation_and_review_workflow():
    source = pd.read_csv(
        APP_PATH.parent / "samples" / "additional-examples" / "sample_products.csv"
    ).fillna("")
    document = make_document("products", "sample_products.csv", source)
    document["target_languages"] = ["English", "Japanese"]
    document["selected_columns"] = ["product_name", "category"]
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"products": document}
    app.session_state["active_document_id"] = "products"
    app.run(timeout=15)

    assert not app.exception
    assert not app.session_state["documents"]["products"]["game_mode"]
    assert not any(button.key == "open_bible_upload_products" for button in app.button)
    assert any(
        button.key == "preview_translation_plan_products" for button in app.button
    )
    assert any(button.key == "translate_products" for button in app.button)
    next(
        button for button in app.button
        if button.key == "preview_translation_plan_products"
    ).click().run(timeout=15)
    assert app.session_state["documents"]["products"]["messages"][-1]["content"].startswith(
        "Estimate ready: **26 translations**"
    )


def test_glossary_workspace_card_only_appears_after_rules_exist():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    document = make_document("one", "first.csv", dataframe)
    document.update({
        "glossary_text": "會員 | English | member",
        "glossary_ui_version": 2,
        "workspace_item_order": ["translation", "glossary"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": document}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    assert not app.exception
    assert any(
        markdown.value.startswith("**Terminology rules**") for markdown in app.markdown
    )
    assert any(button.key == "open_glossary_one" for button in app.button)


def test_translation_upload_rejects_character_bible_without_changing_project():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    document = make_document("one", "first.csv", dataframe)
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": document}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    next(
        button for button in app.button
        if button.key == "open_translation_upload_one"
    ).click().run(timeout=15)
    uploader = next(
        uploader for uploader in app.get("file_uploader")
        if uploader.key.startswith("conversation_dialogue_upload_one_")
    )
    uploader.upload(
        "wrong_bible.csv",
        "speaker,personality,speaking_style\nLuna,Proud,Formal\n".encode("utf-8"),
        "text/csv",
    ).run(timeout=15)

    current = app.session_state["documents"]["one"]
    assert current["name"] == "first.csv"
    assert current["dataframe"].equals(dataframe)
    assert current["messages"][-1]["content"].startswith(
        "This file looks like a Character Bible"
    )


def test_other_conversation_action_cancels_glossary_without_losing_history():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    document = make_document("one", "first.csv", dataframe)
    document["messages"].append({"role": "user", "content": "Keep this earlier message"})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": document}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    next(
        button for button in app.button
        if button.key == "open_conversation_glossary_one"
    ).click().run(timeout=15)
    assert app.session_state["documents"]["one"]["conversation_action"] == "glossary"

    next(
        button for button in app.button
        if button.key == "open_conversation_languages_one"
    ).click().run(timeout=15)
    messages = app.session_state["documents"]["one"]["messages"]
    assert app.session_state["documents"]["one"]["conversation_action"] == "languages"
    assert any(message["content"] == "Keep this earlier message" for message in messages)
    assert messages[-1]["content"] == (
        "Glossary editing cancelled. No terminology rules were changed."
    )


def test_upload_requests_are_persistent_conversation_messages():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {
        "one": make_document("one", "first.csv", dataframe),
    }
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    next(
        button for button in app.button
        if button.key == "open_translation_upload_one"
    ).click().run(timeout=15)
    document = app.session_state["documents"]["one"]
    assert document["messages"][-1]["content"] == (
        "Please upload the translation-data CSV you want to open."
    )
    assert any(
        uploader.key.startswith("conversation_dialogue_upload_one_")
        for uploader in app.get("file_uploader")
    )


def test_updating_translation_data_stays_in_project_and_preserves_setup():
    dialogue = pd.DataFrame({
        "line_id": ["L1"],
        "speaker": ["Luna"],
        "source_text": ["舊台詞。"],
    })
    bible = pd.DataFrame({"speaker": ["Luna"], "personality": ["Calm"]})
    config = GameConfig(scene_id="", listener="", emotion="", context="", character_limit="")
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None), max_retries=0
    ).run(dialogue, config, bible, ["English"])
    document = make_document("game", "old.csv", dialogue)
    document.update({
        "game_mode": True,
        "game_config": result.game_config,
        "character_bible": bible,
        "character_bible_name": "bible.csv",
        "bible_revision": 1,
        "glossary_text": "月石 | English | Moonstone",
        "glossary_draft": "月石 | English | Moonstone",
        "glossary_revision": 1,
        "glossary_ui_version": 2,
        "target_languages": ["English"],
        "result": result,
        "result_target_languages": ["English"],
        "translation_upload_hash": "old-hash",
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": document}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)

    next(
        button for button in app.button if button.key == "open_translation_upload_game"
    ).click().run(timeout=15)
    uploader = next(
        uploader for uploader in app.get("file_uploader")
        if uploader.key.startswith("conversation_dialogue_upload_game_")
    )
    uploader.upload(
        "updated.csv",
        "line_id,speaker,source_text\nL2,Luna,新台詞。\n".encode("utf-8"),
        "text/csv",
    ).run(timeout=15)

    updated = app.session_state["documents"]["game"]
    assert list(app.session_state["documents"]) == ["game"]
    assert app.session_state["active_document_id"] == "game"
    assert updated["name"] == "updated.csv"
    assert updated["dataframe"].iloc[0]["source_text"] == "新台詞。"
    assert updated["target_languages"] == ["English"]
    assert updated["glossary_text"] == "月石 | English | Moonstone"
    assert updated["character_bible"].iloc[0]["personality"] == "Calm"
    assert updated["result"] is None
    assert updated["messages"][-1]["content"].startswith(
        "I replaced this project's source package"
    )


def test_onboarding_can_start_with_character_bible_upload():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    next(button for button in app.button if button.key == "open_onboarding_bible_upload").click().run(timeout=15)
    assert not app.exception
    assert any(
        message["content"].startswith("Please upload a Character Bible CSV")
        for message in app.session_state["onboarding_messages"]
    )
    assert any(
        uploader.label == "Character Bible CSV"
        for uploader in app.get("file_uploader")
    )


def test_pending_character_bible_replaces_blank_workspace():
    bible = pd.DataFrame({
        "speaker": ["Luna", "Player"],
        "personality": ["Bold and direct", "Warm and sincere"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["pending_character_bible"] = bible
    app.session_state["pending_character_bible_name"] = "character_bible.csv"
    app.session_state["pending_character_bible_hash"] = "test-hash"
    app.session_state["pending_workspace_item_order"] = ["bible"]
    app.run(timeout=15)

    assert not app.exception
    assert not any(subheader.value == "Workspace" for subheader in app.subheader)
    assert any(
        "**Character Bible**" in markdown.value and "character_bible.csv" in markdown.value
        for markdown in app.markdown
    )
    assert any(button.key == "open_pending_character_bible" for button in app.button)
    assert any(
        button.key == "open_onboarding_bible_upload" and button.label == "Update Bible"
        for button in app.button
    )


def test_pending_glossary_changes_onboarding_action_to_update():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["onboarding_glossary_text"] = "星核 | English | Astral Core"
    app.session_state["pending_workspace_item_order"] = ["glossary"]
    app.run(timeout=15)

    assert not app.exception
    assert any(
        button.key == "open_onboarding_glossary" and button.label == "Update glossary"
        for button in app.button
    )


def test_workspace_preserves_glossary_bible_file_upload_order():
    dialogue = pd.DataFrame({
        "line_id": ["L1"],
        "speaker": ["Luna"],
        "source_text": ["你好。"],
    })
    bible = pd.DataFrame({"speaker": ["Luna"], "personality": ["Calm"]})
    document = make_document("game", "dialogue.csv", dialogue)
    document.update({
        "glossary_text": "星核 | English | Astral Core",
        "glossary_revision": 1,
        "glossary_ui_version": 2,
        "character_bible": bible,
        "character_bible_name": "bible.csv",
        "bible_revision": 1,
        "workspace_item_order": ["glossary", "bible", "translation"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": document}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)

    assert not app.exception
    markdown_values = [markdown.value for markdown in app.markdown]
    file_position = next(i for i, value in enumerate(markdown_values) if "**Active file**" in value)
    bible_position = next(i for i, value in enumerate(markdown_values) if "**Character Bible**" in value)
    glossary_position = next(i for i, value in enumerate(markdown_values) if "**Terminology rules**" in value)
    assert file_position < bible_position < glossary_position
    assert any(button.label == "Update Bible" for button in app.button)
    assert any(button.label == "Update glossary" for button in app.button)


def test_onboarding_composer_saves_instructions_before_any_upload():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    next(chat for chat in app.chat_input if chat.key == "onboarding_chat_input").set_value(
        "Use concise fantasy dialogue."
    ).run(timeout=15)

    assert not app.exception
    assert app.session_state["onboarding_instructions"] == "Use concise fantasy dialogue."
    assert any(
        message["role"] == "user" and message["content"] == "Use concise fantasy dialogue."
        for message in app.session_state["onboarding_messages"]
    )


def test_file_menu_switches_independent_workspaces():
    first = pd.DataFrame({"id": [1], "title": ["第一份"]})
    second = pd.DataFrame({"id": [2], "title": ["第二份"]})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {
        "one": make_document("one", "first.csv", first),
        "two": make_document("two", "second.csv", second),
    }
    app.session_state["documents"]["two"]["unread"] = True
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    assert not app.exception
    file_buttons = [button for button in app.button if button.key.startswith("open_document_")]
    assert [button.label for button in file_buttons] == ["first", "🔵 second"]
    assert all(button.proto.type == "secondary" for button in file_buttons)
    assert not any(subheader.value == "Conversation" for subheader in app.subheader)
    assert not any(header.value == "first" for header in app.subheader)
    assert not any(subheader.value == "Workspace" for subheader in app.subheader)
    assert any(
        "**Active file**" in markdown.value and "first.csv" in markdown.value
        for markdown in app.markdown
    )
    assert app.chat_message[0].markdown[0].value == "first.csv conversation"

    next(button for button in app.button if button.key == "open_document_two").click().run(timeout=15)

    assert not app.exception
    assert app.session_state["active_document_id"] == "two"
    assert not app.session_state["documents"]["two"]["unread"]
    file_buttons = [button for button in app.button if button.key.startswith("open_document_")]
    assert [button.label for button in file_buttons] == ["first", "second"]
    assert not any(subheader.value == "Conversation" for subheader in app.subheader)
    assert not any(header.value == "second" for header in app.subheader)
    assert not any(subheader.value == "Workspace" for subheader in app.subheader)
    assert any(
        "**Active file**" in markdown.value and "second.csv" in markdown.value
        for markdown in app.markdown
    )
    assert app.chat_message[0].markdown[0].value == "second.csv conversation"
    next(
        button for button in app.button if button.key == "open_translation_setup_two"
    ).click().run(timeout=15)
    assert any(field.key == "selected_columns_two" for field in app.get("multiselect"))


def test_language_multiselect_can_translate_without_typing_a_command():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {
        "one": make_document("one", "first.csv", dataframe),
    }
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    next(button for button in app.button if button.key == "open_conversation_languages_one").click().run(timeout=15)
    language_picker = next(
        field for field in app.get("button_group")
        if field.key == "conversation_languages_one"
    )
    language_picker.set_value(["English", "Japanese"]).run(timeout=15)
    assert next(
        button for button in app.button
        if button.key == "open_conversation_languages_one"
    ).proto.type == "secondary"
    next(button for button in app.button if button.key == "translate_one").click().run(timeout=15)

    assert not app.exception
    assert any(
        message["role"] == "user" and message["content"] == "Translate to English and Japanese"
        for message in app.session_state["documents"]["one"]["messages"]
    )


def test_leaving_language_picker_saves_selection_before_next_action():
    dataframe = pd.DataFrame({"title": ["第一份"]})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {
        "one": make_document("one", "first.csv", dataframe),
    }
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    next(
        button for button in app.button
        if button.key == "open_conversation_languages_one"
    ).click().run(timeout=15)
    next(
        field for field in app.get("button_group")
        if field.key == "conversation_languages_one"
    ).set_value(["English", "Japanese"]).run(timeout=15)
    next(
        button for button in app.button
        if button.key == "open_conversation_glossary_one"
    ).click().run(timeout=15)

    document = app.session_state["documents"]["one"]
    assert document["target_languages"] == ["English", "Japanese"]
    assert document["conversation_action"] == "glossary"
    assert document["messages"][-2:] == [
        {"role": "user", "content": "Target languages: **English, Japanese**"},
        {"role": "assistant", "content": "Target languages saved: **English, Japanese**."},
    ]
    assert app.chat_message[-1].markdown[0].value.startswith(
        "Upload a glossary CSV, or enter terminology rules"
    )
    assert any(
        uploader.label == "Glossary CSV" for uploader in app.get("file_uploader")
    )


def test_sent_instruction_history_is_restored_after_switching_files():
    first = pd.DataFrame({"title": ["第一份"]})
    second = pd.DataFrame({"title": ["第二份"]})
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {
        "one": make_document("one", "first.csv", first),
        "two": make_document("two", "second.csv", second),
    }
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    next(chat for chat in app.chat_input if chat.key == "chat_input_one").set_value(
        "Translate this later"
    ).run(timeout=15)
    assert app.chat_message[-1].markdown[0].value.startswith("Optional instructions saved")
    assert any(button.key == "chat_open_bible_one" for button in app.button) is False
    assert any(button.key == "open_conversation_languages_one" for button in app.button)
    next(button for button in app.button if button.key == "open_document_two").click().run(timeout=15)

    assert app.session_state["documents"]["one"]["translation_instructions"] == "Translate this later"
    assert any(chat.key == "chat_input_two" for chat in app.chat_input)

    next(button for button in app.button if button.key == "open_document_one").click().run(timeout=15)

    assert app.session_state["active_document_id"] == "one"
    assert any(
        message["role"] == "user" and message["content"] == "Translate this later"
        for message in app.session_state["documents"]["one"]["messages"]
    )


def test_completed_result_renders_quality_review_retry_and_execution_history():
    source = pd.DataFrame({"title": ["第一份", "第二份"]})
    translated = source.assign(title__english=["First", "第二份"])
    metrics = TranslationMetrics(
        rows=2,
        source_columns=1,
        target_languages=1,
        requested_unique_values=2,
        translated_unique_values=1,
        failed_unique_values=1,
    )
    failures = pd.DataFrame([{
        "language": "English", "column": "title", "source": "第二份", "error": "test failure"
    }])
    document = make_document("one", "first.csv", source)
    document["result"] = TranslationResult(translated, metrics, failures, ["title"], ["English"])
    document["runs"] = [{
        "Started (UTC)": "2026-01-01T00:00:00+00:00", "Finished (UTC)": "2026-01-01T00:00:01+00:00",
        "Outcome": "Completed", "Mode": "translation", "Model": "test", "Languages": "English",
        "Columns": "title", "Glossary": 0, "Est. tokens": 10, "Est. cost (USD)": 0.001,
        "API calls": 1, "Coverage": 0.5, "Duration (s)": 1.0, "Detail": "",
    }]

    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": document}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    assert not app.exception
    retry_button = next(button for button in app.button if button.label == "Targeted rerun")
    assert not retry_button.disabled
    assert not any(button.label == "Rerun with current glossary" for button in app.button)
    failure_apply = next(button for button in app.button if button.label == "Apply resolution")
    assert failure_apply.disabled
    next(button for button in app.button if button.key == "open_conversation_glossary_one").click().run(timeout=15)
    assert not any(button.key == "cancel_glossary_one" for button in app.button)
    assert any(uploader.label == "Glossary CSV" for uploader in app.get("file_uploader"))
    next(chat for chat in app.chat_input if chat.key == "glossary_chat_input_one").set_value(
        "會員 | English | member"
    ).run(timeout=15)
    assert next(
        button for button in app.button if button.label == "Targeted rerun"
    ).proto.type == "primary"
    assert any(
        message["content"].startswith("Applied **1 terminology rule(s)**")
        for message in app.session_state["documents"]["one"]["messages"]
    )
    assert app.session_state["documents"]["one"]["glossary_revision"] == 1
    assert app.session_state["documents"]["one"]["result_glossary_revision"] == 0
    assert any(expander.label == "Execution history (1)" for expander in app.expander)
    assert any(expander.label == "Failure triage (1 unresolved)" for expander in app.expander)
    failure_apply = next(button for button in app.button if button.label == "Apply resolution")
    assert failure_apply.disabled
    assert app.session_state["documents"]["one"]["failure_triage"].loc[0, "Resolution note"] == "Choose a reason"
    assert list(
        app.session_state["documents"]["one"]["failure_triage"].columns[
            :len(REVIEW_REPORT_COLUMNS_FOR_TEST)
        ]
    ) == REVIEW_REPORT_COLUMNS_FOR_TEST
    review_button = next(
        button for button in app.button if button.key == "open_generic_review_one"
    )
    assert review_button.label == "Open localization review"
    assert review_button.proto.type == "primary"
    review_button.click().run(timeout=15)
    assert any(
        markdown.value == "### Localization review workbench"
        for markdown in app.markdown
    )
    assert any(button.label == "Download CSV" for button in app.get("download_button"))
    assert any(expander.label == "Quality spot check" for expander in app.expander)
    assert len(app.radio) > 0


def failure_document() -> dict:
    source = pd.DataFrame({"title": ["第二份"]})
    translated = source.assign(title__english=["第二份"])
    metrics = TranslationMetrics(
        rows=1,
        source_columns=1,
        target_languages=1,
        requested_unique_values=1,
        translated_unique_values=0,
        failed_unique_values=1,
    )
    failures = pd.DataFrame([{
        "language": "English", "column": "title", "source": "第二份", "error": "test failure"
    }])
    document = make_document("one", "first.csv", source)
    document["result"] = TranslationResult(translated, metrics, failures, ["title"], ["English"])
    return document


def set_failure_input(
    app: AppTest,
    manual_translation: str = "",
    resolution_note: str = "Choose a reason",
) -> None:
    document = app.session_state["documents"]["one"]
    table = document["failure_triage"].copy()
    table.loc[0, "Final translation"] = manual_translation
    table.loc[0, "Resolution note"] = resolution_note
    document["failure_triage"] = table
    document["failure_triage_version"] += 1


def test_guidance_change_requires_a_manually_selected_resolution_note():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": failure_document()}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert apply_button.disabled

    app.session_state["documents"]["one"]["glossary_revision"] += 1
    app.run(timeout=15)

    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert apply_button.disabled
    assert any(
        warning.value == "Choose a Resolution note for every change you want to apply."
        for warning in app.warning
    )

    set_failure_input(app, resolution_note="Terminology corrected")
    app.run(timeout=15)
    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert not apply_button.disabled
    assert any(success.value == "1 resolution(s) ready to apply." for success in app.success)


def test_skip_source_unlocks_apply_and_resolves_without_translation():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": failure_document()}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    set_failure_input(app, resolution_note="Source text kept intentionally")
    app.run(timeout=15)
    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert not apply_button.disabled
    assert app.session_state["documents"]["one"]["failure_triage"].loc[0, "Resolution note"] == "Source text kept intentionally"
    apply_button.click().run(timeout=15)

    document = app.session_state["documents"]["one"]
    assert document["result"].failures.empty
    assert document["result"].dataframe.loc[0, "title__english"] == "第二份"
    assert document["failure_resolutions"][0]["Action"] == "Skip"
    assert any(expander.label == "Failure resolution history (1)" for expander in app.expander)


def test_manual_translation_and_manually_selected_note_unlock_apply():
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": failure_document()}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert apply_button.disabled

    set_failure_input(app, manual_translation="Second item")
    app.run(timeout=15)
    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert apply_button.disabled

    set_failure_input(
        app,
        manual_translation="Second item",
        resolution_note="Translated manually by reviewer",
    )
    app.run(timeout=15)
    apply_button = next(button for button in app.button if button.label == "Apply resolution")
    assert not apply_button.disabled
    assert app.session_state["documents"]["one"]["failure_triage"].loc[0, "Resolution note"] == "Translated manually by reviewer"
    apply_button.click().run(timeout=15)

    document = app.session_state["documents"]["one"]
    assert document["result"].failures.empty
    assert document["result"].dataframe.loc[0, "title__english"] == "Second item"
    assert document["failure_resolutions"][0]["Note"] == "Translated manually by reviewer"


def test_retry_control_stays_visible_and_disabled_without_failures():
    source = pd.DataFrame({"title": ["第一份"]})
    translated = source.assign(title__english=["First"])
    metrics = TranslationMetrics(
        rows=1,
        source_columns=1,
        target_languages=1,
        requested_unique_values=1,
        translated_unique_values=1,
    )
    document = make_document("one", "first.csv", source)
    document["result"] = TranslationResult(
        translated,
        metrics,
        pd.DataFrame(columns=["language", "column", "source", "error"]),
        ["title"],
        ["English"],
    )

    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"one": document}
    app.session_state["active_document_id"] = "one"
    app.run(timeout=15)

    retry_button = next(button for button in app.button if button.label == "Targeted rerun")
    assert not retry_button.disabled
    assert not any(button.label == "Rerun with current glossary" for button in app.button)
    assert any(
        message["role"] == "assistant"
        and message["content"] == "✓ No failed values to retry."
        for message in app.session_state["documents"]["one"]["messages"]
    )


def test_rerun_highlights_and_preselects_only_inputs_changed_since_translation():
    dialogue = pd.DataFrame({
        "line_id": ["L1"],
        "scene_id": ["S1"],
        "speaker": ["Luna"],
        "emotion": ["calm"],
        "source_text": ["你好。"],
    })
    config = GameConfig()
    bible = default_character_bible(dialogue, config)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None),
        max_retries=0,
    ).run(dialogue, config, bible, ["English"])
    document = make_document("game", "dialogue.csv", dialogue)
    document.update({
        "game_mode": True,
        "game_config": result.game_config,
        "character_bible": bible,
        "result": result,
        "review_table": build_review_table(dialogue, result),
        "target_languages": ["English", "Japanese"],
        "result_target_languages": ["English"],
        "bible_revision": 2,
        "result_bible_revision": 1,
        "glossary_revision": 3,
        "result_glossary_revision": 2,
    })

    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": document}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)

    rerun = next(button for button in app.button if button.key == "open_targeted_rerun_game")
    assert rerun.proto.type == "primary"
    rerun.click().run(timeout=15)

    updated_inputs = next(
        field for field in app.get("button_group")
        if field.key.startswith("conversation_rerun_updates_game_")
    )
    assert updated_inputs.options == [
        "Character Bible updated",
        "Target languages updated",
        "Glossary updated",
    ]
    assert updated_inputs.value == updated_inputs.options


def test_game_dialogue_schema_opens_chat_character_bible_and_context_controls():
    dialogue = pd.DataFrame({
        "line_id": ["L1", "L2"],
        "scene_id": ["S1", "S1"],
        "speaker": ["Luna", "Player"],
        "emotion": ["angry", "calm"],
        "source_text": ["你遲到了！", "對不起。"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": make_document("game", "dialogue.csv", dialogue)}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)

    assert not app.exception
    assert app.session_state["documents"]["game"]["game_mode"]
    assert app.session_state["documents"]["game"]["messages"][-1]["content"].startswith(
        "Game localization mode"
    )
    assert set(app.session_state["documents"]["game"]["character_bible"]["speaker"]) == {"Luna", "Player"}
    assert not any(expander.label == "Translation setup" for expander in app.expander)
    next(
        button for button in app.button if button.key == "open_translation_setup_game"
    ).click().run(timeout=15)
    assert any(checkbox.label == "Run AI style evaluation after translation" for checkbox in app.checkbox)
    assert {field.label for field in app.number_input} >= {"Previous lines", "Next lines"}
    assert any(markdown.value == "**Scene & speaker work units**" for markdown in app.markdown)
    assert not any(markdown.value == "**Pre-run estimate**" for markdown in app.markdown)
    assert not any(button.label == "Preview workload & cost" for button in app.button)
    assert "Character guidance is missing" in app.session_state["documents"]["game"]["messages"][-1]["content"]
    assert any("Advanced column mapping" in markdown.value for markdown in app.markdown)
    assert any("Context & style evaluation" in markdown.value for markdown in app.markdown)
    assert not any(field.key == "project_name_game" for field in app.text_input)
    assert any(button.key == "open_bible_upload_game" for button in app.button)
    bible_button = next(
        button for button in app.button if button.key == "open_bible_upload_game"
    )
    assert bible_button.proto.type == "primary"
    app.session_state["documents"]["game"]["bible_revision"] = 1
    app.session_state["documents"]["game"]["character_bible_name"] = "dialogue_bible.csv"
    app.run(timeout=15)
    assert next(
        button for button in app.button if button.key == "open_bible_upload_game"
    ).proto.type == "secondary"
    assert any(
        "**Character Bible**" in markdown.value and "dialogue_bible.csv" in markdown.value
        for markdown in app.markdown
    )
    open_bible = next(
        button for button in app.button if button.key == "open_character_bible_game"
    )
    assert not app.get("file_uploader")
    open_bible.click().run(timeout=15)
    assert not app.exception
    assert any(button.key == "close_bible_game" for button in app.button)
    assert any(button.label == "Save Character Bible" for button in app.button)
    assert any(
        caption.value.startswith("Add one row per speaker") for caption in app.caption
    )
    assert any(button.key == "open_conversation_glossary_game" for button in app.button)


def test_key_source_string_package_works_without_a_speaker_column():
    strings = pd.DataFrame({
        "key": ["btn_confirm", "btn_open"],
        "source": ["Confirm", "Open"],
        "zh-TW": ["", ""],
        "context": ["Main confirmation button", "Chest interaction"],
        "char_limit": [4, 4],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"strings": make_document("strings", "strings.csv", strings)}
    app.session_state["active_document_id"] = "strings"
    app.run(timeout=15)

    assert not app.exception
    assert app.session_state["documents"]["strings"]["game_mode"]
    next(
        button for button in app.button if button.key == "open_translation_setup_strings"
    ).click().run(timeout=15)
    speaker = next(field for field in app.selectbox if field.key == "game_speaker_column_strings")
    source = next(field for field in app.selectbox if field.key == "game_source_column_strings")
    assert speaker.value == "(none)"
    assert source.value == "source"
    bible_button = next(
        button for button in app.button if button.key == "open_bible_upload_strings"
    )
    assert bible_button.disabled


def test_system_speaker_does_not_require_character_guidance():
    strings = pd.DataFrame({
        "key": ["btn_confirm", "dlg_mira_03"],
        "source": ["Confirm", "You'll regret this."],
        "speaker": ["System", "Mira"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"strings": make_document("strings", "strings.csv", strings)}
    app.session_state["active_document_id"] = "strings"
    app.run(timeout=15)

    status_messages = [
        message["content"] for message in app.session_state["documents"]["strings"]["messages"]
        if "Character guidance is missing" in message["content"]
    ]
    assert status_messages
    assert "1 speaker(s): Mira" in status_messages[-1]
    assert "System" not in status_messages[-1]


def test_game_pre_run_estimate_uses_language_multiselect_without_a_message():
    dialogue = pd.DataFrame({
        "line_id": ["L1", "L2"],
        "scene_id": ["S1", "S1"],
        "speaker": ["Luna", "Player"],
        "source_text": ["你遲到了！", "對不起。"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": make_document("game", "dialogue.csv", dialogue)}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)
    assert not any(button.key == "preview_game_plan_game" for button in app.button)
    assert not any(button.key == "translate_game" for button in app.button)
    bible = app.session_state["documents"]["game"]["character_bible"].copy()
    bible["personality"] = "Defined character voice"
    app.session_state["documents"]["game"]["character_bible"] = bible
    app.run(timeout=15)
    assert not any(button.key == "preview_game_plan_game" for button in app.button)
    assert not any(button.key == "translate_game" for button in app.button)
    next(button for button in app.button if button.key == "open_conversation_languages_game").click().run(timeout=15)
    next(
        field for field in app.get("button_group")
        if field.key == "conversation_languages_game"
    ).set_value(["English", "Japanese"]).run(timeout=15)
    assert any(button.key == "preview_game_plan_game" for button in app.button)
    assert any(button.key == "translate_game" for button in app.button)
    app.session_state["documents"]["game"]["conversation_action"] = "rerun"
    app.run(timeout=15)
    next(button for button in app.button if button.key == "preview_game_plan_game").click().run(timeout=15)

    assert not app.exception
    messages = app.session_state["documents"]["game"]["messages"]
    assert messages[-2]["content"] == "Targeted rerun cancelled before it started."
    assert messages[-1]["content"].startswith("Estimate ready: **4 translations**")
    assert app.session_state["documents"]["game"]["conversation_action"] == ""
    assert any(
        message["role"] == "assistant"
        and message["content"].startswith("Estimate ready: **4 translations**")
        for message in app.session_state["documents"]["game"]["messages"]
    )
    assert not any(success.value.startswith("Estimate ready:") for success in app.success)


def test_translate_stays_hidden_while_character_bible_is_incomplete():
    dialogue = pd.DataFrame({
        "line_id": ["L1", "L2"],
        "scene_id": ["S1", "S1"],
        "speaker": ["Luna", "Player"],
        "source_text": ["你遲到了！", "對不起。"],
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": make_document("game", "dialogue.csv", dialogue)}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)
    next(chat for chat in app.chat_input if chat.key == "chat_input_game").set_value(
        "Translate to English"
    ).run(timeout=15)

    assert not app.exception
    assert any(
        "Character guidance is missing for 2 speaker(s): Luna, Player" in message["content"]
        for message in app.session_state["documents"]["game"]["messages"]
    )
    assert not any(button.key == "translate_game" for button in app.button)
    assert app.session_state["documents"]["game"]["translation_instructions"] == "Translate to English"


def test_completed_game_result_renders_review_download_qa_and_targeted_rerun():
    dialogue = pd.DataFrame({
        "line_id": ["L1", "L2"],
        "scene_id": ["S1", "S1"],
        "speaker": ["Luna", "Player"],
        "emotion": ["angry", "calm"],
        "source_text": ["你遲到了！", "對不起。"],
        "character_limit": [50, 50],
    })
    config = GameConfig()
    bible = default_character_bible(dialogue, config)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None),
        max_retries=0,
    ).run(dialogue, config, bible, ["English"], evaluate_style=True)
    document = make_document("game", "dialogue.csv", dialogue)
    document.update({
        "game_mode": True,
        "game_config": result.game_config,
        "character_bible": bible,
        "result": result,
        "target_languages": ["English"],
        "result_target_languages": ["English"],
    })
    review = build_review_table(dialogue, result)
    review.loc[0, "selected"] = True
    document["review_table"] = review
    document["review_grid_version"] = 2
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": document}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)

    assert not app.exception
    assert any(button.label == "Upload Character Bible" for button in app.button)
    assert any(button.label == "Add language" for button in app.button)
    assert any(button.label == "Add glossary" for button in app.button)
    assert any(
        button.key == "open_translation_upload_game"
        and button.label == "Update translation data"
        for button in app.button
    )
    assert not any(button.key == "preview_game_plan_game" for button in app.button)
    assert not any(button.key == "translate_game" for button in app.button)
    assert any(expander.label.startswith("Developer question batch") for expander in app.expander)
    assert any(expander.label.startswith("Game QA issues") for expander in app.expander)
    assert not any(selectbox.key == "game_rerun_scope_game" for selectbox in app.selectbox)
    assert not any(
        button.label == "Download reviewed localization CSV"
        for button in app.get("download_button")
    )
    assert not any(button.key.startswith("review_directory_game_") for button in app.button)
    next(button for button in app.button if button.key == "open_targeted_rerun_game").click().run(timeout=15)
    assert not any(
        field.key == "conversation_rerun_scope_game"
        for field in app.get("button_group")
    )
    assert any(
        "Open localization review for failed values, characters, scenes, context, or QA reruns."
        in caption.value
        for caption in app.caption
    )
    review_button = next(button for button in app.button if button.key == "open_game_review_game")
    assert review_button.label == "Open localization review"
    review_button.click().run(timeout=15)

    assert not app.exception
    assert any(
        markdown.value == "### Localization review workbench"
        for markdown in app.markdown
    )
    assert not any(
        dataframe.key == "review_grid_editor_game_queue_context_notes"
        for dataframe in app.dataframe
    )
    assert any(
        "Story context order keeps neighboring dialogue interleaved" in caption.value
        for caption in app.caption
    )
    assert not any(selectbox.key == "review_mode_game" for selectbox in app.selectbox)
    assert not any(selectbox.key == "review_sort_game" for selectbox in app.selectbox)
    assert not any(button.label.startswith("Rerun scope") for button in app.button)
    assert any(button.label == "Download CSV" for button in app.get("download_button"))
    assert any(button.key == "open_targeted_rerun_game" for button in app.button)
    assert not any(selectbox.key == "inspect_context_game" for selectbox in app.selectbox)
    assert not any(checkbox.label == "Select" for checkbox in app.checkbox)
    assert not any(expander.label.startswith("Context inspector") for expander in app.expander)
    assert any(expander.label.startswith("Developer question batch") for expander in app.expander)
    assert not any(selectbox.key.startswith("translation_card_language_game_") for selectbox in app.selectbox)
    assert any("Translation cards:" in caption.value for caption in app.caption)
    assert not any(button.label == "Save clarification & rerun line" for button in app.button)
    assert not any(button.label == "Add confirmed clarification" for button in app.button)
    assert not any(button.label == "Save review" for button in app.button)
    assert not any(button.label == "Needs developer context" for button in app.button)
    assert not any(button.label == "Keep source text" for button in app.button)
    assert any(expander.label.startswith("Game QA issues") for expander in app.expander)
    assert any(expander.label.startswith("AI style evaluation") for expander in app.expander)


def test_review_selects_first_row_by_default_for_details():
    dialogue = pd.DataFrame({
        "line_id": ["L1"],
        "scene_id": ["S1"],
        "speaker": ["Luna"],
        "source_text": ["打開它。"],
    })
    config = GameConfig()
    bible = default_character_bible(dialogue, config)
    result = GameLocalizationAgent(
        DemoTranslationBackend(delay_seconds=0, failure_marker=None),
        max_retries=0,
    ).run(dialogue, config, bible, ["English"])
    document = make_document("game", "dialogue.csv", dialogue)
    document.update({
        "game_mode": True,
        "game_config": result.game_config,
        "character_bible": bible,
        "result": result,
        "review_table": build_review_table(dialogue, result),
    })
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    app.session_state["documents"] = {"game": document}
    app.session_state["active_document_id"] = "game"
    app.run(timeout=15)
    next(button for button in app.button if button.key == "open_game_review_game").click().run(timeout=15)

    assert not app.exception
    saved_review = app.session_state["documents"]["game"]["review_table"]
    assert int(saved_review["selected"].fillna(False).sum()) == 1
    assert bool(saved_review.iloc[0]["selected"])
    assert not any(subheader.value.startswith("Context inspector") for subheader in app.subheader)
    assert not any(button.label == "Save review" for button in app.button)
    assert not any(selectbox.key == "review_mode_game" for selectbox in app.selectbox)
