from __future__ import annotations

import time

import pandas as pd

import app as app_module
from game_localization import GameConfig, default_character_bible


class FakeBackend:
    def translate(self, texts, target_language, *, model=None):
        time.sleep(0.02)
        return [f"{target_language}: {text}" for text in texts]


def test_background_job_returns_immediately_and_completes_independently(monkeypatch):
    monkeypatch.setattr(app_module, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        app_module,
        "OpenAITranslationBackend",
        lambda client, model: FakeBackend(),
    )
    manager = app_module.BackgroundJobManager(max_workers=1)
    dataframe = pd.DataFrame({"text": ["第一筆", "第二筆", "第三筆"]})

    started = time.monotonic()
    job_id = manager.submit(dataframe, ["text"], ["English"], "test-key")
    submit_elapsed = time.monotonic() - started

    assert submit_elapsed < 0.2
    while manager.status(job_id)["state"] != "completed":
        time.sleep(0.01)
    result = manager.consume(job_id)

    assert result.metrics.coverage == 1.0
    assert result.dataframe["text__english"].tolist() == [
        "English: 第一筆",
        "English: 第二筆",
        "English: 第三筆",
    ]


def test_demo_mode_background_job_uses_no_openai_client(monkeypatch):
    monkeypatch.setenv("TRANSLATION_DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_TRANSLATION_DELAY_SECONDS", "0")

    def openai_must_not_be_called(**kwargs):
        raise AssertionError("OpenAI client was created in Demo Mode")

    monkeypatch.setattr(app_module, "OpenAI", openai_must_not_be_called)
    manager = app_module.BackgroundJobManager(max_workers=1)
    dataframe = pd.DataFrame({"text": ["第一筆"]})
    job_id = manager.submit(dataframe, ["text"], ["English"], "unused-demo-key")
    while manager.status(job_id)["state"] != "completed":
        time.sleep(0.01)
    status = manager.status(job_id)
    result = manager.consume(job_id)

    assert status["metadata"]["model"] == "demo-no-api"
    assert result.dataframe.loc[0, "text__english"] == "第一筆"


def test_demo_targeted_retry_automatically_recovers_marked_failure(monkeypatch):
    monkeypatch.setenv("TRANSLATION_DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_TRANSLATION_DELAY_SECONDS", "0")
    monkeypatch.setenv("TRANSLATION_MAX_RETRIES", "0")
    manager = app_module.BackgroundJobManager(max_workers=1)
    dataframe = pd.DataFrame({"text": ["正常", "[FAIL] 模擬失敗"]})

    first_job = manager.submit(dataframe, ["text"], ["English"], "unused-demo-key")
    while manager.status(first_job)["state"] != "completed":
        time.sleep(0.01)
    first = manager.consume(first_job)

    assert len(first.failures) == 1
    assert first.metrics.coverage == 0.5

    retry_job = manager.submit(
        dataframe,
        ["text"],
        ["English"],
        "unused-demo-key",
        previous_result=first,
    )
    while manager.status(retry_job)["state"] != "completed":
        time.sleep(0.01)
    retried = manager.consume(retry_job)

    assert retried.failures.empty
    assert retried.metrics.coverage == 1.0
    assert retried.dataframe.loc[1, "text__english"] == "[FAIL] 模擬失敗"


def test_background_manager_runs_game_context_and_style_pipeline(monkeypatch):
    monkeypatch.setenv("TRANSLATION_DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_TRANSLATION_DELAY_SECONDS", "0")
    dataframe = pd.DataFrame({
        "line_id": ["L1"],
        "scene_id": ["S1"],
        "speaker": ["Luna"],
        "source_text": ["你遲到了！"],
    })
    config = GameConfig()
    manager = app_module.BackgroundJobManager(max_workers=1)
    job_id = manager.submit(
        dataframe,
        ["source_text"],
        ["English"],
        "unused-demo-key",
        game_config=config,
        character_bible=default_character_bible(dataframe, config),
        evaluate_style=True,
    )
    while manager.status(job_id)["state"] != "completed":
        time.sleep(0.01)
    result = manager.consume(job_id)

    assert result.game_config["source_text"] == "source_text"
    assert result.dataframe.loc[0, "source_text__english"] == "你遲到了！"
    assert "untranslated_chinese" not in set(result.qa_issues["type"])
    assert len(result.style_evaluations) == 1
