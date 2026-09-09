"""Context-aware mobile-game localization and human review utilities."""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import pandas as pd

from translation_agent import (
    CHINESE_RE,
    PLACEHOLDER_RE,
    GlossaryEntry,
    DemoTranslationBackend,
    ProviderTranslationError,
    TranslationCancelled,
    TranslationMetrics,
    TranslationResult,
    detect_source_language,
    glossary_replacements,
    protect_tokens,
    restore_tokens,
    safe_column_suffix,
    text_similarity,
)


RICH_TEXT_RE = re.compile(r"</?[^<>]+>|\[(?:/?(?:b|i|u|color|size|sprite))[^]]*\]", re.I)
AMBIGUOUS_SHORT_RE = re.compile(
    r"^(?:open|close|use|take|go|stop|fine|this|that|it|"
    r"打開|關閉|使用|拿走|走吧|住手|好吧|這個|那個|它)[。.!！?？…]*$",
    re.I,
)
UNCLEAR_REFERENT_RE = re.compile(r"\b(?:it|this|that|they|them|he|she)\b|(?:這個|那個|它|他|她|他們|她們)", re.I)
GAME_REQUIRED_COLUMNS = {"speaker", "source_text"}
CHARACTER_BIBLE_COLUMNS = [
    "speaker",
    "personality",
    "speaking_style",
    "relationship",
    "preferred_tone",
    "avoid",
    "catchphrases",
    "pronouns",
]


@dataclass(frozen=True)
class GameConfig:
    line_id: str = "line_id"
    scene_id: str = "scene_id"
    speaker: str = "speaker"
    source_text: str = "source_text"
    listener: str = "listener"
    emotion: str = "emotion"
    context: str = "context"
    character_limit: str = "character_limit"
    screenshot: str = "screenshot_reference"
    previous_lines: int = 2
    next_lines: int = 1

    def available(self, df: pd.DataFrame, field: str) -> str | None:
        column = getattr(self, field)
        return column if column and column in df.columns else None


def detect_game_schema(df: pd.DataFrame) -> bool:
    config = infer_game_config(df)
    return bool(config.source_text and (config.line_id or config.speaker))


def infer_game_config(df: pd.DataFrame) -> GameConfig:
    by_lower = {str(column).lower(): str(column) for column in df.columns}

    def choose(*aliases: str) -> str:
        return next((by_lower[alias] for alias in aliases if alias in by_lower), "")

    return GameConfig(
        line_id=choose("line_id", "dialogue_id", "key", "string_id"),
        scene_id=choose("scene_id", "scene", "chapter_id", "chapter", "screen", "location"),
        speaker=choose("speaker", "character", "actor"),
        source_text=choose("source_text", "source", "dialogue", "text", "en", "english", "ja", "japanese"),
        listener=choose("listener", "addressee", "target_character"),
        emotion=choose("emotion", "mood"),
        context=choose("context", "scene_context", "description", "developer_note", "note"),
        character_limit=choose("character_limit", "char_limit", "max_length", "length_limit"),
        screenshot=choose("screenshot_reference", "screenshot", "screenshot_url", "image"),
    )


def default_character_bible(df: pd.DataFrame, config: GameConfig) -> pd.DataFrame:
    speaker_column = config.available(df, "speaker")
    speakers = [] if not speaker_column else sorted(
        value for value in df[speaker_column].dropna().astype(str).unique() if value.strip()
    )
    return pd.DataFrame(
        [{"speaker": speaker, **{column: "" for column in CHARACTER_BIBLE_COLUMNS[1:]}} for speaker in speakers],
        columns=CHARACTER_BIBLE_COLUMNS,
    )


def validate_character_bible(df: pd.DataFrame) -> list[str]:
    """Validate that an uploaded CSV is a character guide, not another data file."""
    errors: list[str] = []
    if df is None or df.empty:
        return ["The Character Bible file has no data rows."]
    if "speaker" not in df.columns:
        return ["Missing required column: speaker."]
    guidance_columns = [column for column in CHARACTER_BIBLE_COLUMNS[1:] if column in df.columns]
    if not guidance_columns:
        errors.append(
            "Add at least one character guidance column: "
            + ", ".join(CHARACTER_BIBLE_COLUMNS[1:])
            + "."
        )
    speakers = df["speaker"].fillna("").astype(str).str.strip()
    if speakers.eq("").all():
        errors.append("The speaker column has no character names.")
    duplicates = sorted(speakers[speakers.ne("") & speakers.duplicated(keep=False)].unique())
    if duplicates:
        errors.append("Each speaker must appear once. Duplicate speaker(s): " + ", ".join(duplicates) + ".")
    return errors


def character_profiles(bible: pd.DataFrame) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    if bible is None or bible.empty or "speaker" not in bible:
        return profiles
    for _, row in bible.fillna("").iterrows():
        speaker = str(row["speaker"]).strip()
        if not speaker:
            continue
        profiles[speaker] = {
            column: str(row.get(column, "")).strip()
            for column in CHARACTER_BIBLE_COLUMNS[1:]
        }
    return profiles


def _value(df: pd.DataFrame, row_position: int, column: str | None) -> str:
    if not column:
        return ""
    # df.iloc[row_position][column] reconstructs a whole cross-column row Series on
    # every call; with hundreds of lookups per row (one per config field, repeated
    # for neighboring-line context) that dominates review-grid build time for large
    # documents. Indexing the column first (a cheap view) then the row by position
    # with .iat is the same lookup, orders of magnitude faster at this call volume.
    value = df[column].iat[row_position]
    return "" if pd.isna(value) else str(value)


def describe_placeholders(source: str, translation: str) -> str:
    """List each source placeholder and whether the translation preserved it."""
    placeholders = list(dict.fromkeys(PLACEHOLDER_RE.findall(source)))
    if not placeholders:
        return ""
    translation_placeholders = set(PLACEHOLDER_RE.findall(translation))
    return "; ".join(
        f"{token}: preserved" if token in translation_placeholders else f"{token}: MISSING"
        for token in placeholders
    )


def infer_string_id_clues(string_id: str) -> str:
    """Turn a structured key into transparent hints; these remain unverified context."""
    aliases = {
        "btn": "button",
        "dlg": "dialogue",
        "dialog": "dialogue",
        "ui": "user interface",
        "inv": "inventory",
        "char": "character",
        "desc": "description",
        "msg": "message",
        "err": "error",
    }
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(string_id))
    tokens = [token.lower() for token in re.split(r"[^A-Za-z0-9]+", expanded) if token]
    clues = [aliases.get(token, token) for token in tokens if not token.isdigit()]
    return " · ".join(dict.fromkeys(clues))


def build_game_records(
    df: pd.DataFrame,
    config: GameConfig,
    bible: pd.DataFrame,
    row_positions: Sequence[int] | None = None,
    context_overrides: dict[int, str] | None = None,
) -> list[dict]:
    profiles = character_profiles(bible)
    positions = list(range(len(df))) if row_positions is None else list(row_positions)
    scene_column = config.available(df, "scene_id")
    source_column = config.available(df, "source_text")
    speaker_column = config.available(df, "speaker")
    if not source_column:
        raise ValueError("Game mode requires a source-text column.")

    context_overrides = context_overrides or {}
    records: list[dict] = []
    for position in positions:
        scene = _value(df, position, scene_column)
        neighboring_positions = [
            candidate
            for candidate in range(len(df))
            if not scene_column or _value(df, candidate, scene_column) == scene
        ]
        scene_offset = neighboring_positions.index(position)
        previous = neighboring_positions[max(0, scene_offset - config.previous_lines):scene_offset]
        following = neighboring_positions[
            scene_offset + 1:scene_offset + 1 + config.next_lines
        ]
        speaker = _value(df, position, speaker_column)
        line_id = _value(df, position, config.available(df, "line_id")) or str(position + 1)
        records.append({
            "row_position": position,
            "line_id": line_id,
            "id_clues": infer_string_id_clues(line_id),
            "scene_id": scene,
            "speaker": speaker,
            "listener": _value(df, position, config.available(df, "listener")),
            "emotion": _value(df, position, config.available(df, "emotion")),
            "scene_context": _value(df, position, config.available(df, "context")),
            "context_note": str(context_overrides.get(position, "")).strip(),
            "text": _value(df, position, source_column),
            "previous_lines": [
                {
                    "speaker": _value(df, other, speaker_column),
                    "text": _value(df, other, source_column),
                }
                for other in previous
            ],
            "next_lines": [
                {
                    "speaker": _value(df, other, speaker_column),
                    "text": _value(df, other, source_column),
                }
                for other in following
            ],
            "character": profiles.get(speaker, {}),
        })
    return records


def game_context_risks(
    df: pd.DataFrame,
    config: GameConfig,
    context_overrides: dict[int, str] | None = None,
) -> dict[int, str]:
    """Flag obvious context risks without blocking translation or calling a model."""
    records = build_game_records(
        df, config, pd.DataFrame(), context_overrides=context_overrides
    )
    result: dict[int, str] = {}
    for record in records:
        text = record["text"].strip()
        has_explanation = bool(record["context_note"] or record["scene_context"] or record["listener"])
        risks: list[str] = []
        if not record["scene_id"] and not (record["context_note"] or record["scene_context"]):
            risks.append("Scene/context missing")
        if AMBIGUOUS_SHORT_RE.fullmatch(text) and not has_explanation:
            risks.append("Short string may have multiple meanings")
        if UNCLEAR_REFERENT_RE.search(text) and not has_explanation:
            risks.append("Referent may be unclear")
        result[record["row_position"]] = "; ".join(dict.fromkeys(risks))
    return result


def group_game_records(records: Sequence[dict]) -> list[tuple[tuple[str, str], list[dict]]]:
    """Create stable translation work units without losing original row context."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (
            str(record.get("scene_id") or "Unassigned scene"),
            str(record.get("speaker") or "Unknown speaker"),
        )
        grouped.setdefault(key, []).append(record)
    return list(grouped.items())


def game_work_units(df: pd.DataFrame, config: GameConfig) -> pd.DataFrame:
    """Summarize the scene/speaker units used for batching and workload review."""
    records = build_game_records(df, config, pd.DataFrame())
    rows = []
    for (scene, speaker), group in group_game_records(records):
        rows.append({
            "Scene": scene,
            "Speaker": speaker,
            "Lines": len(group),
            "Characters": sum(len(str(record.get("text", ""))) for record in group),
        })
    return pd.DataFrame(rows, columns=["Scene", "Speaker", "Lines", "Characters"])


def count_game_batches(
    df: pd.DataFrame,
    config: GameConfig,
    batch_size: int,
    row_positions: Sequence[int] | None = None,
) -> int:
    positions = list(range(len(df))) if row_positions is None else list(row_positions)
    records = [
        record
        for record in build_game_records(df, config, pd.DataFrame(), positions)
        if record["text"].strip()
    ]
    return sum(math.ceil(len(group) / batch_size) for _, group in group_game_records(records))


def game_cache_key(record: dict, language: str, glossary_version: str = "") -> tuple:
    character = tuple(sorted((record.get("character") or {}).items()))
    return (
        record.get("text", ""),
        record.get("speaker", ""),
        language,
        character,
        record.get("scene_id", ""),
        record.get("listener", ""),
        record.get("emotion", ""),
        record.get("scene_context", ""),
        record.get("context_note", ""),
        record.get("id_clues", ""),
        tuple((line.get("speaker"), line.get("text")) for line in record.get("previous_lines", [])),
        tuple((line.get("speaker"), line.get("text")) for line in record.get("next_lines", [])),
        glossary_version,
    )


def build_game_qa(
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    config: GameConfig,
    languages: Sequence[str],
    glossary: Sequence[GlossaryEntry],
    check_untranslated_chinese: bool = True,
) -> pd.DataFrame:
    issues: list[dict] = []
    source_column = config.source_text
    for position in range(len(df)):
        source = _value(df, position, source_column)
        line_id = _value(df, position, config.available(df, "line_id")) or str(position + 1)
        limit_raw = _value(df, position, config.available(df, "character_limit"))
        try:
            character_limit = int(float(limit_raw)) if limit_raw else None
        except ValueError:
            character_limit = None
        source_placeholders = set(PLACEHOLDER_RE.findall(source))
        source_tags = set(RICH_TEXT_RE.findall(source))
        for language in languages:
            translated_column = f"{source_column}__{safe_column_suffix(language)}"
            if translated_column not in result_df:
                continue
            translation = _value(result_df, position, translated_column)

            def add(issue_type: str, detail: str, severity: str = "warning") -> None:
                issues.append({
                    "row_position": position,
                    "line_id": line_id,
                    "speaker": _value(df, position, config.available(df, "speaker")),
                    "language": language,
                    "type": issue_type,
                    "severity": severity,
                    "detail": detail,
                })

            if not translation.strip():
                add("empty_translation", "Translation is empty.", "error")
            if (
                check_untranslated_chinese
                and language not in {"Traditional Chinese", "Simplified Chinese"}
                and CHINESE_RE.search(translation)
            ):
                add("untranslated_chinese", "Translation still contains Chinese characters.")
            missing_placeholders = source_placeholders - set(PLACEHOLDER_RE.findall(translation))
            if missing_placeholders:
                add("placeholder_mismatch", f"Missing: {', '.join(sorted(missing_placeholders))}", "error")
            missing_tags = source_tags - set(RICH_TEXT_RE.findall(translation))
            if missing_tags:
                add("rich_text_mismatch", f"Missing tags: {', '.join(sorted(missing_tags))}", "error")
            if character_limit is not None and len(translation) > character_limit:
                add("character_limit", f"{len(translation)} / {character_limit} characters.")
            if translation.strip() and re.search(r"[。！？!?…][\]）】』」》”’]*$", source) and not re.search(
                r"[.!?…][\])}\]»”’\"']*$", translation
            ):
                add("punctuation", "Source ends with dialogue punctuation but translation does not.")
            for entry in glossary:
                if (
                    entry.source in source
                    and entry.target_language == language
                    and entry.translation
                    and entry.translation not in translation
                ):
                    add("glossary_violation", f"Expected term: {entry.translation}", "error")
    for language in languages:
        translated_column = f"{config.source_text}__{safe_column_suffix(language)}"
        if translated_column not in result_df:
            continue
        comparison = pd.DataFrame({
            "row_position": range(len(df)),
            "source": df[config.source_text].fillna("").astype(str),
            "translation": result_df[translated_column].fillna("").astype(str),
        })
        for translation, group in comparison.groupby("translation"):
            if translation and group["source"].nunique() > 1:
                for row in group.itertuples():
                    issues.append({
                        "row_position": row.row_position,
                        "line_id": _value(df, row.row_position, config.available(df, "line_id")) or str(row.row_position + 1),
                        "speaker": _value(df, row.row_position, config.available(df, "speaker")),
                        "language": language,
                        "type": "duplicate_target",
                        "severity": "warning",
                        "detail": "Different source lines share this target translation.",
                    })
    return pd.DataFrame(
        issues,
        columns=["row_position", "line_id", "speaker", "language", "type", "severity", "detail"],
    )


def translation_quality_rates(
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    config: GameConfig,
    languages: Sequence[str],
    glossary: Sequence[GlossaryEntry],
    qa: pd.DataFrame,
) -> dict[str, float | None]:
    """Recall across the whole run: of every (row, language) pair where a source
    placeholder/glossary rule applied, what fraction came through intact.

    protect_tokens/restore_tokens (see translation_agent.py) is supposed to make this
    structurally guaranteed rather than a matter of translation quality — this measures
    it anyway, on the real run's own data, as a live check that the guarantee actually
    held rather than just assuming it. Denominator is 0 when nothing in this run ever
    triggered the rule; the rate is then None so the caller can render "n/a" instead of
    a hollow 100%.
    """
    source_column = config.source_text
    placeholder_total = 0
    glossary_total = 0
    for position in range(len(df)):
        source = _value(df, position, source_column)
        has_placeholder = bool(PLACEHOLDER_RE.search(source))
        for language in languages:
            translated_column = f"{source_column}__{safe_column_suffix(language)}"
            if translated_column not in result_df:
                continue
            if has_placeholder:
                placeholder_total += 1
            for entry in glossary:
                if entry.source in source and entry.target_language == language and entry.translation:
                    glossary_total += 1
    placeholder_issues = int((qa["type"] == "placeholder_mismatch").sum()) if not qa.empty else 0
    glossary_issues = int((qa["type"] == "glossary_violation").sum()) if not qa.empty else 0
    return {
        "placeholder_preservation_rate": (
            (placeholder_total - placeholder_issues) / placeholder_total if placeholder_total else None
        ),
        "glossary_compliance_rate": (
            (glossary_total - glossary_issues) / glossary_total if glossary_total else None
        ),
    }


def back_translation_qa(
    backend,
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    config: GameConfig,
    languages: Sequence[str],
    metrics: TranslationMetrics,
    batch_size: int = 20,
    sample_rate: float = 0.2,
    similarity_threshold: float = 0.45,
) -> pd.DataFrame:
    """Sample translated lines and round-trip them back into the detected source language.

    Low similarity between the original source and the round-trip is a cheap, deterministic
    proxy for meaning drift/hallucination. It is a sampled heuristic for triage, not a
    substitute for human review or a labeled-set metric such as COMET/BLEURT.
    """
    issues: list[dict] = []
    if sample_rate <= 0 or df.empty:
        return pd.DataFrame(issues, columns=["row_position", "line_id", "speaker", "language", "type", "severity", "detail"])
    step = max(1, round(1 / sample_rate))
    sampled_positions = list(range(0, len(df), step))
    for language in languages:
        translated_column = f"{config.source_text}__{safe_column_suffix(language)}"
        if translated_column not in result_df:
            continue
        grouped: dict[str, list[tuple[int, str, str]]] = {}
        for position in sampled_positions:
            source = _value(df, position, config.source_text)
            translation = _value(result_df, position, translated_column)
            if not source.strip() or not translation.strip():
                continue
            back_target = detect_source_language(source)
            if back_target == language:
                continue
            grouped.setdefault(back_target, []).append((position, source, translation))
        for back_target, candidates in grouped.items():
            for start in range(0, len(candidates), batch_size):
                chunk = candidates[start:start + batch_size]
                records = [{"text": translation} for _, _, translation in chunk]
                try:
                    metrics.api_calls += 1
                    round_trips = backend.translate_game(records, back_target)
                except Exception as error:
                    metrics.warnings.append(f"Back-translation check skipped for one batch: {error}")
                    continue
                for (position, source, _translation), round_trip in zip(chunk, round_trips):
                    similarity = text_similarity(source, str(round_trip))
                    if similarity < similarity_threshold:
                        issues.append({
                            "row_position": position,
                            "line_id": _value(df, position, config.available(df, "line_id")) or str(position + 1),
                            "speaker": _value(df, position, config.available(df, "speaker")),
                            "language": language,
                            "type": "possible_hallucination",
                            "severity": "warning",
                            "detail": (
                                f"Back-translation similarity {similarity:.0%}: round-trip "
                                f"\"{round_trip}\" vs source \"{source}\"."
                            ),
                        })
    return pd.DataFrame(
        issues,
        columns=["row_position", "line_id", "speaker", "language", "type", "severity", "detail"],
    )


class GameLocalizationAgent:
    def __init__(
        self,
        backend,
        batch_size: int = 20,
        max_retries: int = 1,
        retry_delay_seconds: float = 0.5,
        escalation_model: str | None = None,
    ):
        self.backend = backend
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.escalation_model = escalation_model

    @staticmethod
    def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
        if should_cancel and should_cancel():
            raise TranslationCancelled("Translation cancelled by the user.")

    def _route_model(self, attempt: int, escalate: bool) -> str | None:
        """Escalate to a stronger model on retry, or proactively for deterministic context-risk lines."""
        if not self.escalation_model:
            return None
        return self.escalation_model if (attempt > 0 or escalate) else None

    def _call(
        self,
        records: list[dict],
        language: str,
        metrics: TranslationMetrics,
        should_cancel,
        escalate: bool = False,
    ) -> list[str]:
        last_error: Exception | None = None
        metrics.record_event("translate")
        escalated_this_call = False
        for attempt in range(self.max_retries + 1):
            self._check_cancelled(should_cancel)
            metrics.api_calls += 1
            if attempt:
                metrics.retries += 1
                metrics.record_event("retry")
            model = self._route_model(attempt, escalate)
            if model and not escalated_this_call:
                metrics.escalated_batches += 1
                escalated_this_call = True
            try:
                if hasattr(self.backend, "translate_game"):
                    values = self.backend.translate_game(records, language, model=model)
                else:
                    values = self.backend.translate([record["text"] for record in records], language, model=model)
                if len(values) != len(records):
                    raise ValueError("Game translation count did not match the batch.")
                metrics.record_event("success")
                return list(values)
            except Exception as error:
                last_error = error
                if isinstance(error, ProviderTranslationError) and not error.retryable:
                    break
                if attempt < self.max_retries and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds * (2 ** attempt))
        metrics.record_event("validate_failed")
        if isinstance(last_error, ProviderTranslationError):
            raise last_error
        raise RuntimeError(str(last_error) if last_error else "Game translation failed")

    def _batch(
        self,
        records: list[dict],
        language: str,
        metrics: TranslationMetrics,
        glossary,
        should_cancel,
        escalate: bool = False,
    ):
        replacements = glossary_replacements(glossary, language)
        protected_records: list[dict] = []
        token_maps: list[dict[str, str]] = []
        for record in records:
            rich_text_tokens = {token: token for token in RICH_TEXT_RE.findall(record["text"])}
            protected, tokens = protect_tokens(record["text"], replacements | rich_text_tokens)
            protected_records.append(record | {"text": protected})
            token_maps.append(tokens)
            metrics.preserved_tokens += len(tokens)
        try:
            outputs = self._call(protected_records, language, metrics, should_cancel, escalate=escalate)
            successes = []
            for record, output, tokens in zip(records, outputs, token_maps):
                restored = restore_tokens(output, tokens)
                if any(value not in restored for value in tokens.values()):
                    raise ValueError("A protected game token was altered.")
                successes.append((record, restored))
            return successes, []
        except TranslationCancelled:
            raise
        except ProviderTranslationError:
            raise
        except Exception as error:
            if len(records) > 1:
                metrics.fallback_splits += 1
                metrics.record_event("split")
                midpoint = math.ceil(len(records) / 2)
                left_ok, left_failed = self._batch(records[:midpoint], language, metrics, glossary, should_cancel, escalate=escalate)
                right_ok, right_failed = self._batch(records[midpoint:], language, metrics, glossary, should_cancel, escalate=escalate)
                return left_ok + right_ok, left_failed + right_failed
            metrics.record_event("retain_source")
            return [], [(records[0], str(error))]

    def run(
        self,
        df: pd.DataFrame,
        config: GameConfig,
        bible: pd.DataFrame,
        languages: Sequence[str],
        glossary: Sequence[GlossaryEntry] = (),
        progress: Callable[[int, int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        row_positions: Sequence[int] | None = None,
        base_result: TranslationResult | None = None,
        approved_locks: set[tuple[int, str]] | None = None,
        evaluate_style: bool = False,
        evaluate_hallucination: bool = False,
        hallucination_sample_rate: float = 0.2,
        hallucination_similarity_threshold: float = 0.45,
        context_overrides: dict[int, str] | None = None,
    ) -> TranslationResult:
        started = time.monotonic()
        approved_locks = approved_locks or set()
        positions = list(range(len(df))) if row_positions is None else sorted(set(int(value) for value in row_positions))
        records = build_game_records(
            df, config, bible, positions, context_overrides=context_overrides
        )
        risks = game_context_risks(df, config, context_overrides)
        output = base_result.dataframe.copy(deep=True) if base_result is not None else df.copy(deep=True)
        metrics = TranslationMetrics(len(df), 1, len(languages))
        failures: list[dict] = []
        style_rows: list[dict] = []

        jobs: list[tuple[str, tuple[str, str], list[dict]]] = []
        for language in languages:
            eligible = [
                record for record in records
                if record["text"].strip() and (record["row_position"], language) not in approved_locks
            ]
            metrics.requested_unique_values += len(eligible)
            translated_column = f"{config.source_text}__{safe_column_suffix(language)}"
            if translated_column not in output:
                output[translated_column] = df[config.source_text].fillna("").astype(str)
            for unit, unit_records in group_game_records(eligible):
                jobs.append((language, unit, unit_records))

        total_batches = sum(math.ceil(len(records) / self.batch_size) for _, _, records in jobs)
        completed = 0
        style_candidates: list[tuple[str, dict, str]] = []
        for language, (scene, speaker), language_records in jobs:
            translated_column = f"{config.source_text}__{safe_column_suffix(language)}"
            for start in range(0, len(language_records), self.batch_size):
                self._check_cancelled(should_cancel)
                batch = language_records[start:start + self.batch_size]
                escalate = any(risks.get(record["row_position"]) for record in batch)
                succeeded, failed = self._batch(batch, language, metrics, glossary, should_cancel, escalate=escalate)
                for record, translation in succeeded:
                    output.at[output.index[record["row_position"]], translated_column] = translation
                    metrics.translated_unique_values += 1
                    style_candidates.append((language, record, translation))
                for record, error in failed:
                    metrics.failed_unique_values += 1
                    failures.append({
                        "row_position": record["row_position"],
                        "line_id": record["line_id"],
                        "speaker": record["speaker"],
                        "language": language,
                        "column": config.source_text,
                        "source": record["text"],
                        "error": error,
                    })
                completed += 1
                if progress:
                    progress(completed, total_batches, f"{language} · {scene} · {speaker}")

        if evaluate_style and hasattr(self.backend, "evaluate_game_style"):
            for start in range(0, len(style_candidates), self.batch_size):
                chunk = style_candidates[start:start + self.batch_size]
                grouped: dict[str, list[tuple[dict, str]]] = {}
                for language, record, translation in chunk:
                    grouped.setdefault(language, []).append((record, translation))
                for language, values in grouped.items():
                    try:
                        metrics.api_calls += 1
                        evaluations = self.backend.evaluate_game_style(
                            [record for record, _ in values],
                            [translation for _, translation in values],
                            language,
                        )
                        for (record, _), evaluation in zip(values, evaluations):
                            style_rows.append({
                                "row_position": record["row_position"],
                                "line_id": record["line_id"],
                                "speaker": record["speaker"],
                                "language": language,
                                "score": int(evaluation.get("score", 0)),
                                "reason": str(evaluation.get("reason", "")),
                                "suggestion": str(evaluation.get("suggestion", "")),
                            })
                    except Exception as error:
                        metrics.warnings.append(f"Style evaluation skipped for one batch: {error}")

        metrics.duration_seconds = time.monotonic() - started
        if failures:
            metrics.warnings.append(f"{len(failures)} game line(s) retain source text after failure.")
        failure_df = pd.DataFrame(failures)
        qa = build_game_qa(
            df,
            output,
            config,
            languages,
            glossary,
            check_untranslated_chinese=not isinstance(self.backend, DemoTranslationBackend),
        )
        if (
            evaluate_hallucination
            and hasattr(self.backend, "translate_game")
            and not isinstance(self.backend, DemoTranslationBackend)
        ):
            hallucination_issues = back_translation_qa(
                self.backend,
                df,
                output,
                config,
                languages,
                metrics,
                batch_size=self.batch_size,
                sample_rate=hallucination_sample_rate,
                similarity_threshold=hallucination_similarity_threshold,
            )
            if not hallucination_issues.empty:
                qa = pd.concat([qa, hallucination_issues], ignore_index=True)
        rates = translation_quality_rates(df, output, config, languages, glossary, qa)
        metrics.placeholder_preservation_rate = rates["placeholder_preservation_rate"]
        metrics.glossary_compliance_rate = rates["glossary_compliance_rate"]
        style_df = pd.DataFrame(
            style_rows,
            columns=["row_position", "line_id", "speaker", "language", "score", "reason", "suggestion"],
        )
        return TranslationResult(
            output,
            metrics,
            failure_df,
            [config.source_text],
            list(languages),
            qa,
            style_df,
            asdict(config),
        )


def build_review_table(
    source_df: pd.DataFrame,
    result: TranslationResult,
    previous: pd.DataFrame | None = None,
    context_overrides: dict[int, str] | None = None,
) -> pd.DataFrame:
    config = GameConfig(**result.game_config)
    context_overrides = context_overrides or {}
    records = {
        record["row_position"]: record
        for record in build_game_records(
            source_df, config, pd.DataFrame(), context_overrides=context_overrides
        )
    }
    context_risks = game_context_risks(source_df, config, context_overrides)
    qa_lookup: dict[tuple[int, str], str] = {}
    qa_issue_lookup: dict[tuple[int, str], str] = {}
    qa_fix_lookup: dict[tuple[int, str], str] = {}
    if not result.qa_issues.empty:
        for key, group in result.qa_issues.groupby(["row_position", "language"]):
            qa_lookup[key] = "; ".join(sorted(group["type"].unique()))
            qa_issue_lookup[key] = "; ".join(
                f"{row.type}: {row.detail}" for row in group.itertuples()
            )
            position, language = int(key[0]), str(key[1])
            translated_column = f"{config.source_text}__{safe_column_suffix(language)}"
            translation = _value(result.dataframe, position, translated_column)
            issue_types = set(group["type"].astype(str))
            if "punctuation" in issue_types and translation.strip():
                source = _value(source_df, position, config.source_text)
                ending = "?" if re.search(r"[？?][\]）】』」》”’]*$", source) else (
                    "!" if re.search(r"[！!][\]）】』」》”’]*$", source) else "."
                )
                qa_fix_lookup[key] = translation.rstrip(".!?") + ending
            else:
                qa_fix_lookup[key] = "No safe automatic fix available"
    style_lookup = {}
    if not result.style_evaluations.empty:
        style_lookup = {
            (int(row.row_position), str(row.language)): row
            for row in result.style_evaluations.itertuples()
        }
    rows: list[dict] = []
    for position in range(len(source_df)):
        for language in result.target_languages:
            translated_column = f"{config.source_text}__{safe_column_suffix(language)}"
            style = style_lookup.get((position, language))
            record = records[position]
            source_value = _value(source_df, position, config.source_text)
            ai_value = _value(result.dataframe, position, translated_column)
            rows.append({
                "selected": False,
                "rerun_selected": False,
                "row_position": position,
                "line_id": _value(source_df, position, config.available(source_df, "line_id")) or str(position + 1),
                "id_clues": record["id_clues"],
                "scene_id": _value(source_df, position, config.available(source_df, "scene_id")),
                "speaker": _value(source_df, position, config.available(source_df, "speaker")),
                "listener": record["listener"],
                "emotion": _value(source_df, position, config.available(source_df, "emotion")),
                "scene_context": record["scene_context"],
                "context_note": record["context_note"],
                "previous_lines": "\n".join(
                    f"{line['speaker']}: {line['text']}" for line in record["previous_lines"]
                ),
                "next_lines": "\n".join(
                    f"{line['speaker']}: {line['text']}" for line in record["next_lines"]
                ),
                "context_risk": context_risks.get(position, ""),
                "character_limit": _value(source_df, position, config.available(source_df, "character_limit")),
                "screenshot": _value(source_df, position, config.available(source_df, "screenshot")),
                "placeholder_details": describe_placeholders(source_value, ai_value),
                "language": language,
                "source_text": source_value,
                "ai_translation": ai_value,
                "reviewed_translation": ai_value,
                "status": "Unreviewed",
                "reviewer_comment": "",
                "qa_flags": qa_lookup.get((position, language), ""),
                "qa_issue": qa_issue_lookup.get((position, language), ""),
                "suggested_fix": qa_fix_lookup.get((position, language), ""),
                "style_score": getattr(style, "score", None),
                "style_reason": getattr(style, "reason", ""),
            })
    table = pd.DataFrame(rows)
    if previous is not None and not previous.empty:
        previous_lookup = {
            (int(row.row_position), str(row.language)): row
            for row in previous.itertuples()
        }
        for index, row in table.iterrows():
            old = previous_lookup.get((int(row["row_position"]), str(row["language"])))
            if old:
                table.at[index, "selected"] = bool(getattr(old, "selected", False))
                table.at[index, "rerun_selected"] = bool(
                    getattr(old, "rerun_selected", False)
                )
                table.at[index, "status"] = (
                    "Unreviewed" if old.status == "Needs revision" else old.status
                )
                table.at[index, "reviewer_comment"] = old.reviewer_comment
                if str(getattr(old, "reviewed_translation", "")).strip():
                    table.at[index, "reviewed_translation"] = old.reviewed_translation
    return table


def approved_locks(review_table: pd.DataFrame | None) -> set[tuple[int, str]]:
    if review_table is None or review_table.empty:
        return set()
    approved = review_table[
        review_table["status"].isin({"Approved", "Keep source text"})
    ]
    return {
        (int(row.row_position), str(row.language))
        for row in approved.itertuples()
    }


def select_review_rows(
    review_table: pd.DataFrame,
    scope: str,
    value: str = "",
    glossary_terms: Sequence[str] = (),
) -> list[int]:
    if review_table is None or review_table.empty:
        return []
    selected = review_table
    if scope == "Selected lines":
        selected = selected[selected["selected"].fillna(False)]
    elif scope == "Character":
        selected = selected[selected["speaker"].eq(value)]
    elif scope == "Scene":
        selected = selected[selected["scene_id"].eq(value)]
    elif scope == "Unreviewed":
        selected = selected[selected["status"].eq("Unreviewed")]
    elif scope == "Needs revision":
        selected = selected[selected["status"].eq("Needs revision")]
    elif scope == "Needs context":
        selected = selected[selected["status"].eq("Needs context")]
    elif scope == "QA flagged":
        selected = selected[selected["qa_flags"].fillna("").ne("")]
    elif scope == "Glossary affected":
        pattern = "|".join(re.escape(term) for term in glossary_terms if term)
        selected = selected[
            selected["source_text"].astype(str).str.contains(pattern, regex=True)
        ] if pattern else selected.iloc[0:0]
    return sorted(set(selected["row_position"].astype(int)))


def reviewed_export(
    source_df: pd.DataFrame,
    result: TranslationResult,
    review_table: pd.DataFrame,
    glossary: Sequence[GlossaryEntry] = (),
) -> pd.DataFrame:
    config = GameConfig(**result.game_config)
    output = result.dataframe.copy(deep=True)
    cards = build_translation_cards(source_df, result, review_table, glossary)
    confidence_lookup = {
        (int(row.row_position), str(row.language)): int(row.confidence)
        for row in cards.itertuples()
    }
    for language in result.target_languages:
        suffix = safe_column_suffix(language)
        reviewed_column = f"{config.source_text}__{suffix}__reviewed"
        status_column = f"{config.source_text}__{suffix}__status"
        comment_column = f"{config.source_text}__{suffix}__comment"
        notes_column = f"{config.source_text}__{suffix}__notes"
        needs_context_column = f"{config.source_text}__{suffix}__needs_context"
        keep_source_column = f"{config.source_text}__{suffix}__keep_source_text"
        confidence_column = f"{config.source_text}__{suffix}__confidence_percent"
        qa_issue_column = f"{config.source_text}__{suffix}__qa_issue"
        suggested_fix_column = f"{config.source_text}__{suffix}__suggested_fix"
        final_translation_column = f"{config.source_text}__{suffix}__final_translation"
        output[reviewed_column] = ""
        output[status_column] = ""
        output[comment_column] = ""
        output[notes_column] = ""
        output[needs_context_column] = False
        output[keep_source_column] = False
        output[confidence_column] = 0
        output[qa_issue_column] = ""
        output[suggested_fix_column] = ""
        output[final_translation_column] = ""
        rows = review_table[review_table["language"].eq(language)]
        for row in rows.itertuples():
            index = output.index[int(row.row_position)]
            output.at[index, reviewed_column] = row.reviewed_translation or row.ai_translation
            output.at[index, status_column] = row.status
            output.at[index, comment_column] = row.reviewer_comment
            output.at[index, notes_column] = row.reviewer_comment
            output.at[index, needs_context_column] = row.status == "Needs context"
            output.at[index, keep_source_column] = row.status == "Keep source text"
            output.at[index, confidence_column] = confidence_lookup.get(
                (int(row.row_position), str(row.language)), 0
            )
            output.at[index, qa_issue_column] = str(getattr(row, "qa_issue", ""))
            output.at[index, suggested_fix_column] = str(getattr(row, "suggested_fix", ""))
            output.at[index, final_translation_column] = row.reviewed_translation or row.ai_translation
    return output


def clean_export(
    source_df: pd.DataFrame,
    result: TranslationResult,
    review_table: pd.DataFrame,
) -> pd.DataFrame:
    """A shippable export: string id, the original source text, and each language's final
    translation, with no review metadata (status, confidence, QA issue, notes, etc.)."""
    config = GameConfig(**result.game_config)
    output = pd.DataFrame(index=source_df.index)
    id_column = config.available(source_df, "line_id")
    if id_column:
        output[id_column] = source_df[id_column]
    output[config.source_text] = source_df[config.source_text]
    for language in result.target_languages:
        column = f"{config.source_text}__{safe_column_suffix(language)}"
        output[column] = ""
        rows = review_table[review_table["language"].eq(language)]
        for row in rows.itertuples():
            index = output.index[int(row.row_position)]
            output.at[index, column] = row.reviewed_translation or row.ai_translation
    return output


def approved_translation_memory(review_table: pd.DataFrame | None) -> dict[tuple[str, str], str]:
    """Treat reviewer-approved edits as project-local exact-match translation memory."""
    if review_table is None or review_table.empty:
        return {}
    approved = review_table[review_table["status"].eq("Approved")]
    memory: dict[tuple[str, str], str] = {}
    for row in approved.itertuples():
        translation = str(row.reviewed_translation or row.ai_translation).strip()
        if translation:
            memory[(str(row.source_text), str(row.language))] = translation
    return memory


def build_translation_cards(
    source_df: pd.DataFrame,
    result: TranslationResult,
    review_table: pd.DataFrame,
    glossary: Sequence[GlossaryEntry] = (),
    context_overrides: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Combine model output with deterministic evidence, risk, QA, and project TM."""
    config = GameConfig(**result.game_config)
    context_overrides = context_overrides or {}
    records = {
        record["row_position"]: record
        for record in build_game_records(
            source_df, config, pd.DataFrame(), context_overrides=context_overrides
        )
    }
    risks = game_context_risks(source_df, config, context_overrides)
    memory = approved_translation_memory(review_table)
    failed_keys: set[tuple[int, str]] = set()
    if result.failures is not None and not result.failures.empty:
        failed_keys = {
            (int(row.row_position), str(row.language))
            for row in result.failures.itertuples()
            if getattr(row, "row_position", None) is not None
        }
    qa_lookup: dict[tuple[int, str], str] = {}
    if not result.qa_issues.empty:
        for key, group in result.qa_issues.groupby(["row_position", "language"]):
            qa_lookup[(int(key[0]), str(key[1]))] = "; ".join(sorted(group["type"].unique()))

    cards: list[dict] = []
    for review_row in review_table.itertuples():
        position = int(review_row.row_position)
        language = str(review_row.language)
        record = records[position]
        source = str(review_row.source_text)
        glossary_hits = [
            f"{entry.source} → {entry.translation or entry.source}"
            for entry in glossary
            if entry.source in source and (entry.target_language in {None, language})
        ]
        sources: list[str] = []
        if record["scene_context"]:
            sources.append(f"Developer note: {record['scene_context']}")
        if record["context_note"]:
            sources.append(f"Verified context: {record['context_note']}")
        if record["id_clues"]:
            sources.append(f"ID inference (unverified): {record['id_clues']}")
        if record["scene_id"]:
            sources.append(f"Location: {record['scene_id']}")
        previous_count = len(record["previous_lines"])
        next_count = len(record["next_lines"])
        if previous_count or next_count:
            sources.append(
                f"Dialogue context: {previous_count} previous line(s) · "
                f"{next_count} next line(s)"
            )
        elif record["scene_id"]:
            sources.append("Related strings: None found")
        # Glossary and TM hits are surfaced as their own Details fields (glossary_hits,
        # tm_match) rather than folded into this prose summary, to avoid showing the
        # same fact twice in the compact panel.
        tm_translation = memory.get((source, language), "")

        risk = risks.get(position, "")
        qa_items = [item for item in qa_lookup.get((position, language), "").split("; ") if item]
        suggested = str(review_row.ai_translation)
        tm_reliable = bool(tm_translation and "different contexts" not in risk)
        if tm_reliable and suggested != tm_translation:
            qa_items.append("approved_tm_mismatch")
        qa = "; ".join(dict.fromkeys(qa_items))
        score = 50
        score += 20 if record["scene_context"] or record["context_note"] else 0
        score += 10 if record["listener"] or record["emotion"] else 0
        score += 10 if glossary_hits else 0
        score += 25 if tm_reliable else 0
        score -= 20 * len([item for item in risk.split("; ") if item])
        score -= 15 if qa else 0
        score = max(5, min(100, score))
        level = "High" if score >= 80 else "Medium" if score >= 55 else "Low"
        if (position, language) in failed_keys:
            score = 0
            level = "Failed"
        cards.append({
            "row_position": position,
            "line_id": str(review_row.line_id),
            "language": language,
            "source": source,
            "suggested_translation": suggested,
            "reviewed_translation": str(review_row.reviewed_translation),
            "review_status": str(review_row.status),
            "reviewer_comment": str(review_row.reviewer_comment),
            "confidence": score,
            "confidence_level": level,
            "context_sources": "\n".join(sources) or "No context evidence available",
            "ambiguity": risk,
            "provisional_note": (
                "Current translation is provisional; confirm the intended meaning."
                if risk else ""
            ),
            "qa_flags": qa,
            "tm_match": tm_translation,
            "glossary_hits": "; ".join(glossary_hits),
        })
    return pd.DataFrame(cards)


def build_batch_context_questions(cards: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate red-flag ambiguities into a compact developer question list."""
    if cards is None or cards.empty:
        return pd.DataFrame(columns=["question_id", "topic", "affected_string_ids", "question"])
    risky = cards[
        cards["ambiguity"].fillna("").ne("")
        | cards["review_status"].eq("Needs context")
    ].copy()
    def question_topic(row: pd.Series) -> str:
        ambiguity = str(row.get("ambiguity") or "").strip()
        if ambiguity:
            return ambiguity
        reviewer_comment = str(row.get("reviewer_comment") or "").strip()
        if reviewer_comment:
            return f"Reviewer request: {reviewer_comment}"
        return f"Reviewer requested context for {row.get('line_id')}"

    risky["question_topic"] = risky.apply(question_topic, axis=1)
    rows: list[dict] = []
    for index, (topic, group) in enumerate(risky.groupby("question_topic", sort=True), start=1):
        ids = list(dict.fromkeys(group["line_id"].astype(str)))
        examples = list(dict.fromkeys(group["source"].astype(str)))[:3]
        notes = [
            note for note in dict.fromkeys(group["reviewer_comment"].astype(str))
            if note.strip()
        ]
        rows.append({
            "question_id": f"CTX-{index:03d}",
            "topic": topic,
            "affected_string_ids": ", ".join(ids),
            "question": (
                f"Please clarify the intended scene, referent, or interaction for {', '.join(ids)}. "
                f"Source example(s): {' | '.join(examples)}"
                + (f" Reviewer note(s): {' | '.join(notes[:3])}" if notes else "")
            ),
        })
    return pd.DataFrame(rows, columns=["question_id", "topic", "affected_string_ids", "question"])
