from __future__ import annotations

import pandas as pd

import json

from translation_agent import (
    DemoTranslationBackend,
    OpenAITranslationBackend,
    ProviderTranslationError,
    TranslationAgent,
    detect_chinese_columns,
    detect_source_language,
    parse_languages,
    parse_glossary,
    protect_tokens,
    restore_tokens,
    text_similarity,
    TranslationCancelled,
)


class FakeOpenAIClient:
    """Mimics the OpenAI SDK response shape for a single chat.completions.create call."""

    def __init__(self, content: str, finish_reason: str = "stop"):
        self.chat = self
        self.completions = self
        self._content = content
        self._finish_reason = finish_reason

    def create(self, **kwargs):
        message = type("Message", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message, "finish_reason": self._finish_reason})()
        return type("Response", (), {"choices": [choice]})()


class FakeBackend:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[list[str], str]] = []
        self.fail_on = fail_on

    def translate(self, texts, target_language, *, model=None):
        values = list(texts)
        self.calls.append((values, target_language))
        if self.fail_on and any(self.fail_on in value for value in values):
            raise RuntimeError("simulated provider failure")
        return [f"{target_language}: {value}" for value in values]


class UnavailableBackend:
    def __init__(self):
        self.calls = 0

    def translate(self, texts, target_language, *, model=None):
        self.calls += 1
        raise ProviderTranslationError("service unavailable", retryable=False)


def test_detects_chinese_text_but_skips_metadata_and_low_ratio():
    df = pd.DataFrame({
        "id": ["中文編號", "另一個"],
        "title": ["好吃", "新鮮"],
        "notes": ["English", "偶爾中文"],
        "price": [10, 20],
        "url": ["https://example.com/中文", "https://example.com"],
    })
    assert detect_chinese_columns(df, threshold=0.75) == ["title"]


def test_parses_multiple_languages_from_chat_and_deduplicates():
    assert parse_languages("請翻成日文、英文 and Japanese") == ["Japanese", "English"]
    assert parse_languages("Traditional Chinese and simplified Chinese") == [
        "Traditional Chinese", "Simplified Chinese"
    ]


def test_protects_and_restores_urls_codes_emails_and_placeholders():
    source = "新品 SKU-123 請看 https://example.com/a 或寄信 a@b.com 給 {{name}}，共 {count} 件，玩家 %1$s"
    protected, tokens = protect_tokens(source)
    assert "SKU-123" not in protected
    assert "https://example.com/a" not in protected
    assert "a@b.com" not in protected
    assert "{{name}}" not in protected
    assert "{count}" not in protected
    assert "%1$s" not in protected
    assert restore_tokens(protected, tokens) == source


def test_supports_over_100_rows_batching_and_preserves_originals():
    df = pd.DataFrame({
        "id": range(125),
        "text": [f"第 {index} 筆商品" for index in range(125)],
    })
    backend = FakeBackend()
    result = TranslationAgent(backend, batch_size=20, max_retries=0).run(
        df, ["text"], ["English", "Japanese"]
    )
    assert len(result.dataframe) == 125
    assert result.dataframe[["id", "text"]].equals(df)
    assert list(result.dataframe.columns) == ["id", "text", "text__english", "text__japanese"]
    assert result.metrics.api_calls == 14
    assert result.metrics.coverage == 1.0
    assert result.failures.empty


def test_failed_batch_splits_and_only_failed_value_degrades_to_source():
    df = pd.DataFrame({"text": ["正常一", "BAD 中文", "正常二"]})
    backend = FakeBackend(fail_on="BAD")
    result = TranslationAgent(
        backend, batch_size=3, max_retries=0, retry_delay_seconds=0
    ).run(df, ["text"], ["English"])
    assert result.dataframe.loc[0, "text__english"].startswith("English:")
    assert result.dataframe.loc[1, "text__english"] == "BAD 中文"
    assert result.dataframe.loc[2, "text__english"].startswith("English:")
    assert result.metrics.translated_unique_values == 2
    assert result.metrics.failed_unique_values == 1
    assert result.metrics.coverage == 2 / 3
    assert result.metrics.fallback_splits > 0
    assert len(result.failures) == 1


def test_provider_wide_error_stops_without_recursive_batch_splitting():
    df = pd.DataFrame({"text": ["第一筆", "第二筆", "第三筆"]})
    backend = UnavailableBackend()
    try:
        TranslationAgent(backend, batch_size=3, max_retries=2, retry_delay_seconds=0).run(
            df, ["text"], ["English"]
        )
    except ProviderTranslationError as error:
        assert str(error) == "service unavailable"
    else:
        raise AssertionError("ProviderTranslationError was not raised")
    assert backend.calls == 1


def test_glossary_preserves_terms_and_applies_language_specific_translation():
    entries, errors = parse_glossary("OpenAI\n會員 | English | member\n會員 | Japanese | 会員")
    assert not errors
    df = pd.DataFrame({"text": ["OpenAI 會員服務"]})
    result = TranslationAgent(FakeBackend(), batch_size=10, max_retries=0).run(
        df, ["text"], ["English"], glossary=entries
    )
    translated = result.dataframe.loc[0, "text__english"]
    assert "OpenAI" in translated
    assert "member" in translated


def test_estimate_counts_deduplicated_values_batches_tokens_and_cost():
    df = pd.DataFrame({"text": ["重複", "重複", "不同"]})
    estimate = TranslationAgent(FakeBackend(), batch_size=1).estimate(
        df, ["text"], ["English", "Japanese"]
    )
    assert estimate.unique_values == 4
    assert estimate.batches == 4
    assert estimate.input_tokens > 0
    assert estimate.output_tokens > 0
    assert estimate.estimated_cost_usd > 0


def test_cancellation_stops_before_the_next_api_call():
    backend = FakeBackend()
    agent = TranslationAgent(backend, batch_size=1, max_retries=0)
    try:
        agent.run(pd.DataFrame({"text": ["第一筆"]}), ["text"], ["English"], should_cancel=lambda: True)
    except TranslationCancelled:
        pass
    else:
        raise AssertionError("TranslationCancelled was not raised")
    assert backend.calls == []


def test_retry_failures_translates_only_failed_values_and_merges_result():
    source = pd.DataFrame({"text": ["正常一", "BAD 中文", "正常二"]})
    first = TranslationAgent(FakeBackend(fail_on="BAD"), batch_size=3, max_retries=0).run(
        source, ["text"], ["English"]
    )
    retry_backend = FakeBackend()
    retried = TranslationAgent(retry_backend, batch_size=10, max_retries=0).retry_failures(source, first)
    assert retry_backend.calls == [(["BAD 中文"], "English")]
    assert retried.failures.empty
    assert retried.metrics.coverage == 1.0
    assert retried.dataframe.loc[1, "text__english"].startswith("English:")


def test_demo_backend_is_local_deterministic_and_can_trigger_failures():
    backend = DemoTranslationBackend(delay_seconds=0)
    assert backend.translate(["會員服務"], "English") == ["會員服務"]
    try:
        backend.translate(["[FAIL] 測試"], "English")
    except RuntimeError as error:
        assert "Demo failure" in str(error)
    else:
        raise AssertionError("Demo failure marker did not trigger")


def test_detect_source_language_covers_chinese_japanese_and_english():
    assert detect_source_language("你好，世界") == "Chinese"
    assert detect_source_language("こんにちは") == "Japanese"
    assert detect_source_language("Hello there") == "English"


def test_text_similarity_matches_identical_and_penalizes_divergence():
    assert text_similarity("Hello world", "Hello world") == 1.0
    assert text_similarity("Hello world", "Completely different sentence") < 0.5


def test_model_routing_escalates_for_long_text_and_records_metric():
    class RecordingBackend:
        def __init__(self):
            self.calls: list[tuple[list[str], str | None]] = []

        def translate(self, texts, target_language, *, model=None):
            self.calls.append((list(texts), model))
            return [f"{target_language}: {value}" for value in texts]

    backend = RecordingBackend()
    agent = TranslationAgent(backend, batch_size=10, max_retries=0, escalation_model="gpt-4o", long_text_threshold=5)
    df = pd.DataFrame({"text": ["hi", "a fairly long piece of source text"]})
    result = agent.run(df, ["text"], ["English"])
    assert backend.calls[0][1] == "gpt-4o"
    assert result.metrics.escalated_batches == 1


def test_model_routing_escalates_on_retry_after_a_transient_failure():
    class FlakyBackend:
        def __init__(self):
            self.calls: list[tuple[list[str], str | None]] = []

        def translate(self, texts, target_language, *, model=None):
            self.calls.append((list(texts), model))
            if len(self.calls) == 1:
                raise RuntimeError("transient failure")
            return [f"{target_language}: {value}" for value in texts]

    backend = FlakyBackend()
    agent = TranslationAgent(backend, batch_size=10, max_retries=1, escalation_model="gpt-4o", long_text_threshold=1000)
    df = pd.DataFrame({"text": ["ok"]})
    result = agent.run(df, ["text"], ["English"])
    assert backend.calls[0][1] is None
    assert backend.calls[1][1] == "gpt-4o"
    assert result.metrics.escalated_batches == 1
    assert result.metrics.coverage == 1.0


def test_openai_backend_recovers_translation_when_model_echoes_whole_record():
    raw = json.dumps({
        "0": {
            "row_position": 0,
            "line_id": "context_low_01",
            "speaker": "System",
            "text": "Open it",
        }
    })
    client = FakeOpenAIClient(raw)
    backend = OpenAITranslationBackend(client, model="gpt-4o-mini")
    result = backend.translate_game([{"text": "打開它", "speaker": "System"}], "English")
    assert result == ["Open it"]


def test_openai_backend_still_fails_when_echoed_record_has_no_text_field():
    raw = json.dumps({"0": {"row_position": 0, "speaker": "System"}})
    client = FakeOpenAIClient(raw)
    backend = OpenAITranslationBackend(client, model="gpt-4o-mini")
    try:
        backend.translate_game([{"text": "打開它", "speaker": "System"}], "English")
    except ValueError as error:
        assert "missing game translation" in str(error)
    else:
        raise AssertionError("Expected a ValueError for a response with no recoverable text")


def test_state_events_trace_split_and_retain_source_on_persistent_failure():
    class AlwaysFailBackend:
        def translate(self, texts, target_language, *, model=None):
            raise RuntimeError("boom")

    agent = TranslationAgent(AlwaysFailBackend(), batch_size=10, max_retries=0)
    df = pd.DataFrame({"text": ["a", "b"]})
    result = agent.run(df, ["text"], ["English"])
    assert "split" in result.metrics.state_events
    assert "retain_source" in result.metrics.state_events
    assert result.metrics.failed_unique_values == 2
