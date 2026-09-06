"""Streamlit UI for the Multilingual Translation Agent."""

from __future__ import annotations

import io
import getpass
import hashlib
import importlib
import math
import os
import re
import threading
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import asdict
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from openai import OpenAI

import game_localization as game_localization_module

# Streamlit reruns app.py without automatically reloading imported local modules.
# Reload the game workflow so local rule changes are reflected without restarting the server.
# translation_agent is deliberately NOT reloaded this way: it defines exception classes
# (TranslationCancelled, ProviderTranslationError) that cross module boundaries, and
# importlib.reload() replaces a module's classes with new objects on every rerun, which
# breaks isinstance/except checks holding an older reference. A translation_agent.py edit
# needs a full server restart to take effect; that's the safer trade-off.
game_localization_module = importlib.reload(game_localization_module)

from translation_agent import (
    DemoTranslationBackend,
    GlossaryEntry,
    OpenAITranslationBackend,
    PROMPT_VERSION,
    TranslationAgent,
    TranslationCancelled,
    WorkloadEstimate,
    parse_glossary,
    parse_languages,
    profile_columns,
    safe_column_suffix,
)
from game_localization import (
    CHARACTER_BIBLE_COLUMNS,
    GameConfig,
    GameLocalizationAgent,
    approved_translation_memory,
    approved_locks,
    build_batch_context_questions,
    build_game_qa,
    build_review_table,
    build_translation_cards,
    count_game_batches,
    default_character_bible,
    detect_game_schema,
    game_work_units,
    infer_game_config,
    reviewed_export,
    select_review_rows,
    validate_character_bible,
)


st.set_page_config(page_title="Multilingual Translation Agent", page_icon="🌐", layout="wide")

RESOLUTION_NOTES = [
    "Choose a reason",
    "Translated manually by reviewer",
    "Source text kept intentionally",
    "Character voice adjusted",
    "Terminology corrected",
    "Context clarified",
    "Mistranslation corrected",
    "Placeholder or formatting fixed",
    "Length constraint fixed",
    "Other",
]

REVIEW_REPORT_COLUMNS = [
    "String ID",
    "Language",
    "Speaker",
    "Original",
    "AI translation",
    "Final translation",
    "Developer note",
    "Length",
    "Context review",
    "Needs context",
    "QA issue",
    "Suggested fix",
    "Confidence",
    "Failure reason",
    "Available context and provenance",
    "Resolution note",
]

TARGET_LANGUAGE_OPTIONS = [
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Italian",
    "Traditional Chinese",
    "Simplified Chinese",
]

NON_CHARACTER_SPEAKERS = {
    "system", "ui", "narrator", "narration", "tutorial", "notification", "unknown",
}


REVIEW_GRID_HTML = """
  <div class="review-grid-shell">
  <div class="review-grid-actions">
    <label class="toolbar-select">
      <select class="sort-by" aria-label="Sort by">
        <option value="Story context order">Sort by: Story context order</option>
        <option value="Needs context first">Sort by: Needs context first</option>
        <option value="QA issues first">Sort by: QA issues first</option>
        <option value="Low confidence first">Sort by: Low confidence first</option>
      </select>
    </label>
    <span class="tooltip" title="Story context order preserves dialogue sequence. The other options move context-risk or QA-flagged rows to the top.">?</span>
    <label class="toolbar-select rerun-scope-control">
      <select class="rerun-scope" aria-label="Targeted rerun scope">
        <option value="Selected lines">Rerun: Selected rows</option>
        <option value="Failed values">Rerun: Failed values</option>
        <option value="Character">Rerun: Character</option>
        <option value="Scene">Rerun: Scene</option>
        <option value="Needs context">Rerun: Needs context</option>
        <option value="QA flagged">Rerun: QA flagged</option>
      </select>
    </label>
    <label class="toolbar-select rerun-value-control" hidden>
      <select class="rerun-value" aria-label="Targeted rerun value"></select>
    </label>
    <span class="rerun-status"></span>
    <span class="toolbar-spacer"></span>
    <button class="targeted-rerun" type="button" hidden>Targeted rerun</button>
    <button class="save-review" type="button">Save review changes</button>
    <span class="save-status">All changes saved</span>
  </div>
  <div class="review-grid-legend" aria-label="Review color legend">
    <span><i class="legend-swatch context"></i>Yellow: context review</span>
    <span><i class="legend-swatch qa"></i>Red: QA issue</span>
    <span><i class="legend-swatch low"></i>Blue: low confidence</span>
  </div>
  <div class="review-grid-scroll">
    <table>
      <thead>
        <tr>
          <th class="rerun-column">Select</th>
          <th class="details-column">Details</th>
          <th class="id-column">String ID</th>
          <th class="language-column">Language</th>
          <th class="speaker-column">Speaker</th>
          <th class="text-column">Original</th>
          <th class="text-column">AI translation</th>
          <th class="text-column">Final translation</th>
          <th class="text-column">Developer note</th>
          <th class="length-column">Length</th>
          <th class="context-column">Context review</th>
          <th class="flag-column">Needs context</th>
          <th class="qa-column">QA issue</th>
          <th class="text-column">Suggested fix</th>
          <th class="confidence-column">Confidence</th>
          <th class="failure-column">Failure reason</th>
          <th class="notes-column">Notes</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
  <section class="review-grid-details" hidden>
    <div class="details-heading">
      <strong></strong>
      <button class="close-details" type="button" aria-label="Close details">×</button>
    </div>
    <div class="details-fields"></div>
    <div class="details-provenance"></div>
  </section>
</div>
"""

REVIEW_GRID_CSS = """
:host {
  color: var(--st-text-color, #111827);
  font-family: var(--st-font, sans-serif);
}
.review-grid-shell {
  position: relative;
  height: calc(100dvh - 7.5rem);
  min-height: 480px;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}
.review-grid-scroll {
  min-height: 0;
  flex: 1;
  overflow: auto;
  border: 1px solid var(--st-border-color, #d1d5db);
  border-radius: var(--st-base-radius, 0.5rem);
  background: #ffffff;
}
table {
  width: 100%;
  min-width: 1260px;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.875rem;
}
th {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 0.65rem 0.55rem;
  text-align: left;
  background: #f3f4f6;
  border-bottom: 1px solid #d1d5db;
  white-space: nowrap;
}
td {
  padding: 0.38rem 0.5rem;
  border-bottom: 1px solid #e5e7eb;
  vertical-align: middle;
  background: #ffffff;
}
tr.context-risk td.translation-cell {
  background: #fff3bf;
}
tr.context-risk td.confidence-cell {
  background: #fff3bf;
}
tr.context-risk td.context-review-cell {
  background: #fff3bf;
}
tr.qa-issue td.qa-cell,
tr.qa-issue td.final-translation-cell {
  background: #fee2e2;
}
tr.low-confidence td.confidence-cell {
  color: #1e3a8a;
  background: #dbeafe;
}
tr.locked td {
  color: #6b7280;
  background: #f8fafc;
}
tr.locked td.translation-cell {
  background: #f8fafc;
}
.rerun-column, .details-column { width: 72px; text-align: center; }
.id-column { min-width: 170px; }
.language-column { min-width: 110px; }
.speaker-column { min-width: 120px; }
.confidence-column { min-width: 135px; }
.text-column { min-width: 240px; }
.context-column { min-width: 240px; }
.qa-column { min-width: 240px; }
.failure-column { min-width: 240px; }
.notes-column { min-width: 220px; }
.flag-column { min-width: 130px; text-align: center; }
.length-column { min-width: 90px; text-align: center; }
.length-over { color: #b91c1c; font-weight: 600; }
td.center { text-align: center; }
input[type="text"] {
  width: 100%;
  box-sizing: border-box;
  padding: 0.42rem 0.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 0.35rem;
  color: #111827;
  background: #ffffff;
  font: inherit;
}
input:disabled {
  border-color: transparent;
  color: #6b7280;
  background: transparent;
  cursor: not-allowed;
}
.suggested-fix-cell {
  min-width: 260px;
}
.suggested-fix-layout {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.original-layout {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.original-layout span {
  flex: 1;
  min-width: 0;
  white-space: normal;
}
.suggested-fix-layout span {
  flex: 1;
  white-space: normal;
}
.use-fix {
  flex: 0 0 auto;
  border: 1px solid var(--st-primary-color, #ff4b4b);
  border-radius: 0.35rem;
  padding: 0.3rem 0.5rem;
  color: var(--st-primary-color, #ff4b4b);
  background: #ffffff;
  font-weight: 600;
  cursor: pointer;
}
.use-fix:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.keep-source-toggle.active {
  color: #ffffff;
  background: var(--st-primary-color, #ff4b4b);
}
input[type="checkbox"], input[type="radio"] {
  accent-color: var(--st-primary-color, #ff4b4b);
}
.review-grid-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 0 0 auto;
  padding: 0.1rem 0;
}
.review-grid-legend {
  display: flex;
  align-items: center;
  gap: 1rem;
  color: #4b5563;
  font-size: 0.8rem;
}
.review-grid-legend span {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.legend-swatch {
  display: inline-block;
  width: 0.85rem;
  height: 0.85rem;
  border: 1px solid #d1d5db;
  border-radius: 0.2rem;
}
.legend-swatch.context { background: #fff3bf; }
.legend-swatch.qa { background: #fee2e2; }
.legend-swatch.low { background: #dbeafe; }
.toolbar-select select {
  min-width: 235px;
  padding: 0.48rem 2rem 0.48rem 0.65rem;
  border: 1px solid #cbd5e1;
  border-radius: var(--st-base-radius, 0.5rem);
  color: #111827;
  background: #ffffff;
  font: inherit;
  font-weight: 600;
}
.sort-by { min-width: 205px !important; }
.rerun-scope-control select { min-width: 175px; }
.rerun-value-control select { min-width: 130px; max-width: 170px; }
.tooltip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.1rem;
  height: 1.1rem;
  margin-left: -0.45rem;
  border: 1px solid #9ca3af;
  border-radius: 999px;
  color: #6b7280;
  font-size: 0.72rem;
  cursor: help;
}
.toolbar-spacer { flex: 1; }
.review-grid-actions .save-review,
.review-grid-actions .targeted-rerun {
  border: 1px solid var(--st-primary-color, #ff4b4b);
  border-radius: var(--st-base-radius, 0.5rem);
  padding: 0.52rem 0.9rem;
  color: #ffffff;
  background: var(--st-primary-color, #ff4b4b);
  font-weight: 600;
  cursor: pointer;
}
.review-grid-actions .targeted-rerun[hidden] {
  display: none !important;
}
.review-grid-actions .rerun-value-control[hidden] {
  display: none !important;
}
.review-grid-actions span {
  color: #6b7280;
  font-size: 0.82rem;
}
.review-grid-actions span.unsaved {
  color: #b45309;
  font-weight: 600;
}
.review-grid-actions .rerun-status {
  max-width: 190px;
  line-height: 1.2;
}
.review-grid-details {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 3;
  box-sizing: border-box;
  height: 195px;
  overflow: auto;
  padding: 0.55rem 0.7rem;
  border: 1px solid #d1d5db;
  border-radius: var(--st-base-radius, 0.5rem);
  background: #ffffff;
  box-shadow: 0 -4px 14px rgba(15, 23, 42, 0.12);
}
.review-grid-details[hidden] {
  display: none;
}
.details-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.45rem;
}
.close-details {
  border: 0;
  padding: 0 0.25rem;
  color: #6b7280;
  background: transparent;
  font-size: 1.25rem;
  cursor: pointer;
}
.details-fields {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 0.55rem;
}
.details-field span {
  display: block;
  color: #6b7280;
  font-size: 0.72rem;
  margin-bottom: 0.12rem;
}
.details-field div, .details-provenance {
  font-size: 0.82rem;
  white-space: pre-wrap;
}
.details-provenance {
  margin-top: 0.5rem;
  color: #4b5563;
}
"""

REVIEW_GRID_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const tbody = parentElement.querySelector("tbody")
  const applyButton = parentElement.querySelector(".save-review")
  const targetedRerunButton = parentElement.querySelector(".targeted-rerun")
  const sortBy = parentElement.querySelector(".sort-by")
  const rerunScope = parentElement.querySelector(".rerun-scope")
  const rerunValue = parentElement.querySelector(".rerun-value")
  const rerunValueControl = parentElement.querySelector(".rerun-value-control")
  const rerunStatus = parentElement.querySelector(".rerun-status")
  const saveStatus = parentElement.querySelector(".save-status")
  const detailsPanel = parentElement.querySelector(".review-grid-details")
  if (!tbody || !applyButton) return

  const rows = Array.isArray(data?.rows) ? data.rows : []
  if (sortBy) sortBy.value = data?.sort_by ?? "Story context order"
  const storageKey = `review-grid-draft:${data?.storage_key ?? "default"}`
  let draft = null
  try {
    draft = JSON.parse(window.localStorage.getItem(storageKey) || "null")
  } catch (_) {
    draft = null
  }
  const draftRows = new Map(
    Array.isArray(draft?.rows) ? draft.rows.map((row) => [String(row.row_key), row]) : []
  )
  const submittedDraftMatches = Boolean(draft?.submitted) && rows.every((row) => {
    const saved = draftRows.get(String(row.row_key))
    return !saved || (
      String(saved.translation ?? "") === String(row.translation ?? "") &&
      String(saved.notes ?? "") === String(row.notes ?? "") &&
      Boolean(saved.needs_context) === Boolean(row.needs_context) &&
      Boolean(saved.keep_source) === Boolean(row.keep_source) &&
      Boolean(saved.rerun) === Boolean(row.rerun)
    )
  })
  if (submittedDraftMatches) {
    window.localStorage.removeItem(storageKey)
    draft = null
    draftRows.clear()
  }
  tbody.replaceChildren()

  const collectRows = () => Array.from(tbody.querySelectorAll("tr")).map((tr) => ({
    row_key: tr.dataset.rowKey,
    rerun: Boolean(tr.querySelector('[data-field="rerun"]')?.checked),
    details: Boolean(tr.querySelector('input[type="radio"]')?.checked),
    translation: tr.querySelector('[data-field="translation"]')?.value ?? "",
    notes: tr.querySelector('[data-field="notes"]')?.value ?? "",
    needs_context: Boolean(tr.querySelector('[data-field="needs_context"]')?.checked),
    keep_source: tr.querySelector('[data-field="keep_source"]')?.classList.contains("active") ?? false,
  }))

  const persistDraft = () => {
    window.localStorage.setItem(storageKey, JSON.stringify({ rows: collectRows() }))
    if (saveStatus) {
      saveStatus.textContent = "Unsaved changes"
      saveStatus.classList.add("unsaved")
    }
  }

  const updateTargetedRerunVisibility = () => {
    if (!targetedRerunButton) return
    const scope = rerunScope?.value ?? "Selected lines"
    const selectedCount = collectRows().filter((row) => row.rerun).length
    const available = {
      "Selected lines": selectedCount > 0,
      "Failed values": Boolean(data?.has_failures),
      "Character": Boolean(rerunValue?.value),
      "Scene": Boolean(rerunValue?.value),
      "Needs context": Boolean(data?.has_needs_context),
      "QA flagged": Boolean(data?.has_qa_flagged),
    }
    targetedRerunButton.hidden = !available[scope]
    if (rerunStatus) {
      const messages = {
        "Selected lines": selectedCount
          ? `${selectedCount} row(s) selected`
          : "",
        "Failed values": data?.has_failures
          ? `${data?.failed_count ?? 0} failed value(s) available`
          : "No failed values in this result.",
        "Character": rerunValue?.value ? `Character: ${rerunValue.value}` : "No character available.",
        "Scene": rerunValue?.value ? `Scene: ${rerunValue.value}` : "No scene available.",
        "Needs context": data?.has_needs_context
          ? `${data?.needs_context_count ?? 0} context-risk row(s) available`
          : "No context-risk rows in this result.",
        "QA flagged": data?.has_qa_flagged
          ? `${data?.qa_flagged_count ?? 0} QA-flagged row(s) available`
          : "No QA-flagged rows in this result.",
      }
      rerunStatus.textContent = messages[scope] ?? ""
    }
  }

  const updateRerunValue = () => {
    if (!rerunScope || !rerunValue || !rerunValueControl) return
    const scope = rerunScope.value
    const options = scope === "Character"
      ? (data?.characters ?? [])
      : scope === "Scene" ? (data?.scenes ?? []) : []
    rerunValue.replaceChildren()
    options.forEach((value) => {
      const option = document.createElement("option")
      option.value = value
      option.textContent = value
      rerunValue.appendChild(option)
    })
    rerunValueControl.hidden = !["Character", "Scene"].includes(scope)
    updateTargetedRerunVisibility()
  }

  const showDetails = (row) => {
    if (!detailsPanel) return
    detailsPanel.hidden = false
    detailsPanel.querySelector(".details-heading strong").textContent =
      `Details · ${row.label}`
    const fields = detailsPanel.querySelector(".details-fields")
    fields.replaceChildren()
    ;[
      ["Scene", row.scene],
      ["Speaker → Listener", row.speaker_listener],
      ["Emotion", row.emotion],
      ["Previous dialogue", row.previous_lines],
      ["Next dialogue", row.next_lines],
      ["Glossary hits", row.glossary_hits],
      ["TM matches", row.tm_match],
      ["Placeholder details", row.placeholder_details],
      ["Glossary & protected terms", row.glossary_rules],
    ].forEach(([label, value]) => {
      const field = document.createElement("div")
      field.className = "details-field"
      const fieldLabel = document.createElement("span")
      fieldLabel.textContent = label
      const fieldValue = document.createElement("div")
      fieldValue.textContent = value || "Not provided"
      field.append(fieldLabel, fieldValue)
      fields.appendChild(field)
    })
    const screenshotField = document.createElement("div")
    screenshotField.className = "details-field"
    const screenshotLabel = document.createElement("span")
    screenshotLabel.textContent = "Screenshot"
    const screenshotValue = document.createElement("div")
    if (row.screenshot && (row.screenshot.startsWith("http://") || row.screenshot.startsWith("https://"))) {
      const link = document.createElement("a")
      link.href = row.screenshot
      link.target = "_blank"
      link.rel = "noopener noreferrer"
      link.textContent = "View screenshot"
      screenshotValue.appendChild(link)
    } else {
      screenshotValue.textContent = row.screenshot || "Not provided"
    }
    screenshotField.append(screenshotLabel, screenshotValue)
    fields.appendChild(screenshotField)
    const provenance = detailsPanel.querySelector(".details-provenance")
    provenance.textContent = row.context_sources
      ? `Reconstructed context and provenance: ${row.context_sources}`
      : ""
  }

  const addTextCell = (tr, value, className = "") => {
    const td = document.createElement("td")
    td.className = className
    td.textContent = value ?? ""
    tr.appendChild(td)
  }

  rows.forEach((sourceRow) => {
    const saved = draftRows.get(String(sourceRow.row_key))
    const row = saved ? {
      ...sourceRow,
      rerun: Boolean(saved.rerun),
      details: Boolean(saved.details),
      translation: saved.translation ?? sourceRow.translation,
      notes: saved.notes ?? sourceRow.notes,
      needs_context: Boolean(saved.needs_context),
      keep_source: Boolean(saved.keep_source),
    } : sourceRow
    const tr = document.createElement("tr")
    tr.dataset.rowKey = row.row_key
    if (row.context_risk) tr.classList.add("context-risk")
    if (row.qa_issue) tr.classList.add("qa-issue")
    if (row.low_confidence) tr.classList.add("low-confidence")
    if (!row.editable) tr.classList.add("locked")

    const rerunCell = document.createElement("td")
    rerunCell.className = "center"
    const rerun = document.createElement("input")
    rerun.type = "checkbox"
    rerun.checked = Boolean(row.rerun)
    rerun.dataset.field = "rerun"
    rerun.onchange = () => {
      persistDraft()
      updateTargetedRerunVisibility()
    }
    rerunCell.appendChild(rerun)
    tr.appendChild(rerunCell)

    const detailsCell = document.createElement("td")
    detailsCell.className = "center"
    const details = document.createElement("input")
    details.type = "radio"
    details.name = "review-details"
    details.checked = Boolean(row.details)
    details.onchange = () => {
      showDetails(row)
      persistDraft()
    }
    detailsCell.appendChild(details)
    tr.appendChild(detailsCell)

    addTextCell(tr, row.string_id)
    addTextCell(tr, row.language)
    addTextCell(tr, row.speaker)
    const originalCell = document.createElement("td")
    originalCell.className = "original-cell"
    const originalLayout = document.createElement("div")
    originalLayout.className = "original-layout"
    const keep = document.createElement("button")
    keep.type = "button"
    keep.className = "use-fix keep-source-toggle"
    keep.disabled = !row.editable
    keep.dataset.field = "keep_source"
    const setKeepState = (active) => {
      keep.classList.toggle("active", active)
      keep.textContent = "Keep source"
      keep.setAttribute("aria-pressed", String(active))
    }
    setKeepState(Boolean(row.keep_source))
    const originalText = document.createElement("span")
    originalText.textContent = row.original ?? ""
    originalLayout.appendChild(originalText)
    originalLayout.appendChild(keep)
    originalCell.appendChild(originalLayout)
    tr.appendChild(originalCell)

    addTextCell(tr, row.ai_translation)

    const translationCell = document.createElement("td")
    translationCell.className = "translation-cell final-translation-cell"
    const translation = document.createElement("input")
    translation.type = "text"
    translation.value = row.translation ?? ""
    translation.disabled = !row.editable
    translation.dataset.field = "translation"
    translationCell.appendChild(translation)
    tr.appendChild(translationCell)

    addTextCell(tr, row.developer_note)

    const lengthCell = document.createElement("td")
    lengthCell.className = "length-column"
    const limit = row.character_limit ? Number(row.character_limit) : null
    const updateLength = () => {
      const current = translation.value.length
      lengthCell.textContent = limit ? `${current} / ${limit}` : String(current)
      lengthCell.classList.toggle("length-over", Boolean(limit) && current > limit)
    }
    updateLength()
    translation.oninput = () => {
      persistDraft()
      updateLength()
    }
    tr.appendChild(lengthCell)

    addTextCell(tr, row.context_review, "context-review-cell")

    const needsCell = document.createElement("td")
    needsCell.className = "center"
    const needs = document.createElement("input")
    needs.type = "checkbox"
    needs.checked = Boolean(row.needs_context)
    needs.disabled = !row.editable
    needs.dataset.field = "needs_context"
    needsCell.appendChild(needs)
    tr.appendChild(needsCell)

    addTextCell(tr, row.qa_issue, "qa-cell")

    const suggestedCell = document.createElement("td")
    suggestedCell.className = "suggested-fix-cell"
    const suggestedLayout = document.createElement("div")
    suggestedLayout.className = "suggested-fix-layout"
    const suggestedText = document.createElement("span")
    suggestedText.textContent = row.suggested_fix || ""
    suggestedLayout.appendChild(suggestedText)
    if (row.suggested_fix && row.suggested_fix !== "No safe automatic fix available") {
      const useFix = document.createElement("button")
      useFix.type = "button"
      useFix.className = "use-fix"
      useFix.textContent = "Use fix"
      useFix.disabled = !row.editable
      useFix.onclick = () => {
        translation.value = row.suggested_fix
        persistDraft()
        updateLength()
      }
      suggestedLayout.appendChild(useFix)
    }
    suggestedCell.appendChild(suggestedLayout)
    tr.appendChild(suggestedCell)

    addTextCell(tr, row.confidence, "confidence-cell")
    addTextCell(tr, row.failure_reason)

    const notesCell = document.createElement("td")
    const notes = document.createElement("input")
    notes.type = "text"
    notes.value = row.notes ?? ""
    notes.disabled = !row.editable
    notes.dataset.field = "notes"
    notes.oninput = persistDraft
    notesCell.appendChild(notes)
    tr.appendChild(notesCell)

    needs.onchange = () => {
      if (needs.checked) setKeepState(false)
      persistDraft()
    }
    keep.onclick = () => {
      const next = !keep.classList.contains("active")
      setKeepState(next)
      if (next) needs.checked = false
      persistDraft()
    }
    tbody.appendChild(tr)
    if (details.checked) showDetails(row)
  })
  updateTargetedRerunVisibility()

  applyButton.onclick = () => {
    const submittedRows = collectRows()
    window.localStorage.setItem(
      storageKey,
      JSON.stringify({ rows: submittedRows, submitted: true })
    )
    if (saveStatus) {
      saveStatus.textContent = "Saving…"
      saveStatus.classList.remove("unsaved")
    }
    setTriggerValue("submitted", { rows: submittedRows })
  }

  if (targetedRerunButton) {
    targetedRerunButton.onclick = () => {
      const submittedRows = collectRows()
      setTriggerValue("targeted_rerun", {
        rows: submittedRows,
        scope: rerunScope?.value ?? "Selected lines",
        value: rerunValue?.value ?? "",
      })
    }
  }
  if (rerunScope) rerunScope.onchange = updateRerunValue
  if (rerunValue) rerunValue.onchange = updateTargetedRerunVisibility
  updateRerunValue()

  const changeView = () => {
    persistDraft()
    setTriggerValue("view", {
      sort_by: sortBy?.value ?? "Story context order",
    })
  }
  if (sortBy) sortBy.onchange = changeView

  const closeButton = detailsPanel?.querySelector(".close-details")
  if (closeButton) closeButton.onclick = () => { detailsPanel.hidden = true }

  if (draftRows.size && saveStatus) {
    saveStatus.textContent = draft?.submitted ? "Saved" : "Unsaved draft restored"
    saveStatus.classList.toggle("unsaved", !draft?.submitted)
  }
}
"""

REVIEW_GRID_COMPONENT = st.components.v2.component(
    "review_grid",
    html=REVIEW_GRID_HTML,
    css=REVIEW_GRID_CSS,
    js=REVIEW_GRID_JS,
)


def demo_mode_enabled() -> bool:
    return os.getenv("TRANSLATION_DEMO_MODE", "").lower() in {"1", "true", "yes", "on"}


def hallucination_check_enabled() -> bool:
    return os.getenv("TRANSLATION_HALLUCINATION_CHECK", "").lower() in {"1", "true", "yes", "on"}


def modified_by() -> str:
    """Stand in for a real authenticated identity; this app has no multi-user login."""
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def glossary_signature(entries: list[GlossaryEntry]) -> str:
    canonical = "\n".join(
        f"{entry.source}|{entry.target_language or ''}|{entry.translation or ''}|{entry.term_type}|{entry.notes}"
        for entry in entries
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def project_name_from_file(filename: str) -> str:
    return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip() or "Untitled project"


def estimate_game_workload(
    dataframe: pd.DataFrame,
    config: GameConfig,
    languages: list[str],
    evaluate_style: bool,
    batch_size: int | None = None,
) -> WorkloadEstimate:
    batch_size = batch_size or int(os.getenv("TRANSLATION_BATCH_SIZE", "25"))
    input_rate = float(os.getenv("OPENAI_INPUT_COST_PER_MILLION", "0.15"))
    output_rate = float(os.getenv("OPENAI_OUTPUT_COST_PER_MILLION", "0.60"))
    character_count = int(dataframe[config.source_text].fillna("").astype(str).str.len().sum())
    nonempty_rows = int(dataframe[config.source_text].fillna("").astype(str).str.strip().ne("").sum())
    values = nonempty_rows * len(languages)
    batches = count_game_batches(dataframe, config, batch_size) * len(languages)
    input_tokens = math.ceil(character_count * len(languages) / 4 + batches * 300)
    output_tokens = math.ceil(character_count * len(languages) * 1.2 / 4)
    if evaluate_style:
        input_tokens *= 2
        output_tokens *= 2
    cost = input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
    return WorkloadEstimate(values, batches, input_tokens, output_tokens, cost)


def missing_bible_speakers(document: dict) -> list[str]:
    """Return dialogue speakers that do not yet have any voice guidance."""
    if not document.get("game_mode") or not document.get("game_config"):
        return []
    dataframe = document["dataframe"]
    config = GameConfig(**document["game_config"])
    if not config.speaker or config.speaker not in dataframe.columns:
        return []
    bible = document.get("character_bible")
    if bible is None or bible.empty or "speaker" not in bible.columns:
        guided_speakers: set[str] = set()
    else:
        bible = bible.fillna("")
        guidance_columns = [column for column in bible.columns if column != "speaker"]
        guided_speakers = {
            str(row["speaker"]).strip()
            for _, row in bible.iterrows()
            if str(row.get("speaker", "")).strip()
            and any(str(row.get(column, "")).strip() for column in guidance_columns)
        }
    source_speakers = {
        str(value).strip()
        for value in dataframe[config.speaker].dropna().astype(str)
        if str(value).strip() and str(value).strip().lower() not in NON_CHARACTER_SPEAKERS
    }
    return sorted(source_speakers - guided_speakers)


def touch_workspace_item(document: dict, item: str) -> None:
    """Move an updated workspace section to the newest position."""
    order = [value for value in document.get("workspace_item_order", []) if value != item]
    order.append(item)
    document["workspace_item_order"] = order


def touch_pending_workspace_item(item: str) -> None:
    """Track asset creation order before a translation project exists."""
    order = [
        value for value in st.session_state.get("pending_workspace_item_order", [])
        if value != item
    ]
    order.append(item)
    st.session_state.pending_workspace_item_order = order


def ordered_pending_workspace_items() -> list[str]:
    """Return staged onboarding assets from newest to oldest."""
    available: list[str] = []
    if str(st.session_state.get("onboarding_glossary_text", "")).strip():
        available.append("glossary")
    if st.session_state.get("pending_character_bible") is not None:
        available.append("bible")
    order = [
        item for item in st.session_state.get("pending_workspace_item_order", [])
        if item in available
    ]
    for item in available:
        if item not in order:
            order.append(item)
    st.session_state.pending_workspace_item_order = order
    return list(reversed(order))


def ordered_workspace_items(document: dict) -> list[str]:
    """Return available workspace sections from newest to oldest."""
    available = ["translation"]
    if str(document.get("glossary_text", "")).strip():
        available.append("glossary")
    if document.get("bible_revision", 0) > 0:
        available.append("bible")
    if document.get("result") is not None:
        available.append("result")
    order = [
        item for item in document.get("workspace_item_order", [])
        if item in available
    ]
    for item in available:
        if item not in order:
            order.append(item)
    document["workspace_item_order"] = order
    return list(reversed(order))


def render_workspace_status(document: dict) -> None:
    """Render high-priority game guidance inside the assistant conversation."""
    if document.get("last_job_error"):
        st.error(document["last_job_error"], icon="🚨")
    if not document.get("game_mode"):
        return
    st.success("Game localization mode · string package and available context ready")
    missing_guidance = missing_bible_speakers(document)
    if missing_guidance:
        st.warning(
            f"Character guidance is missing for {len(missing_guidance)} speaker(s): "
            f"{', '.join(missing_guidance)}. Translation can continue, but those speakers will use a neutral voice. "
            "Upload a Character Bible here for character-specific dialogue."
        )
        return

    document["send_bible_warning"] = ""
    document["awaiting_bible_override"] = False
    config = GameConfig(**document["game_config"]) if document.get("game_config") else GameConfig()
    dataframe = document["dataframe"]
    if config.speaker and config.speaker in dataframe.columns:
        speaker_count = (
            dataframe[config.speaker].dropna().astype(str).str.strip().replace("", pd.NA).nunique()
        )
        st.success(f"Character Bible ready for all {speaker_count} speaker(s).")


def game_status_message(document: dict) -> str:
    """Build the current game-readiness update as durable conversation text."""
    if not document.get("game_mode"):
        return ""
    lines = ["Game localization mode · string package and available context ready"]
    missing_guidance = missing_bible_speakers(document)
    if missing_guidance:
        lines.append(
            f"Character guidance is missing for {len(missing_guidance)} speaker(s): "
            f"{', '.join(missing_guidance)}. Translation can continue, but those speakers will "
            "use a neutral voice. Upload a Character Bible for character-specific dialogue."
        )
    else:
        config = GameConfig(**document["game_config"]) if document.get("game_config") else GameConfig()
        if config.speaker and config.speaker in document["dataframe"].columns:
            speaker_count = (
                document["dataframe"][config.speaker]
                .dropna().astype(str).str.strip().replace("", pd.NA).nunique()
            )
            lines.append(f"Character Bible ready for all {speaker_count} speaker(s).")
    return "\n\n".join(lines)


def ensure_game_status_message(document: dict) -> None:
    """Append a readiness update only when its content has actually changed."""
    message = game_status_message(document)
    if not message or message == document.get("game_status_signature"):
        return
    add_message(document, "assistant", message)
    document["game_status_signature"] = message


class BackgroundJobManager:
    """Keep translation work alive while Streamlit rerenders or switches files."""

    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="translation")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit(
        self,
        dataframe: pd.DataFrame,
        source_columns: list[str],
        languages: list[str],
        key: str,
        glossary: list[GlossaryEntry] | None = None,
        previous_result=None,
        glossary_revision: int = 0,
        game_config: GameConfig | None = None,
        character_bible: pd.DataFrame | None = None,
        row_positions: list[int] | None = None,
        game_base_result=None,
        game_approved_locks: set[tuple[int, str]] | None = None,
        evaluate_style: bool = False,
        game_retry_failures: bool = False,
        context_overrides: dict[int, str] | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex
        glossary = glossary or []
        demo_mode = demo_mode_enabled()
        model = "demo-no-api" if demo_mode else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        batch_size = int(os.getenv("TRANSLATION_BATCH_SIZE", "25"))
        input_rate = float(os.getenv("OPENAI_INPUT_COST_PER_MILLION", "0.15"))
        output_rate = float(os.getenv("OPENAI_OUTPUT_COST_PER_MILLION", "0.60"))
        estimator = TranslationAgent(object(), batch_size=batch_size)
        if game_config is not None:
            positions = list(range(len(dataframe))) if row_positions is None else row_positions
            if row_positions is None:
                estimate = estimate_game_workload(dataframe, game_config, languages, evaluate_style, batch_size)
            else:
                subset = dataframe.iloc[positions].reset_index(drop=True)
                estimate = estimate_game_workload(subset, game_config, languages, evaluate_style, batch_size)
            mode = "game targeted rerun" if game_base_result is not None else "game translation"
        elif previous_result is None:
            estimate = estimator.estimate(dataframe, source_columns, languages, input_rate, output_rate)
            mode = "translation"
        else:
            failed = previous_result.failures
            unique_values = len(failed)
            characters = int(failed["source"].astype(str).str.len().sum())
            batches = sum(
                (len(group) + batch_size - 1) // batch_size
                for _, group in failed.groupby(["language", "column"], sort=False)
            )
            input_tokens = int((characters / 4 + batches * 120) + 0.999)
            output_tokens = int((characters * 1.2 / 4) + 0.999)
            cost = input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
            estimate = WorkloadEstimate(unique_values, batches, input_tokens, output_tokens, cost)
            mode = "failed-values retry"
        cancel_event = threading.Event()
        metadata = {
            "job_id": job_id,
            "mode": mode,
            "model": model,
            "languages": ", ".join(languages),
            "columns": ", ".join(source_columns),
            "glossary_entries": len(glossary),
            "glossary_signature": glossary_signature(glossary),
            "glossary_revision": glossary_revision,
            "prompt_version": PROMPT_VERSION,
            "modified_by": modified_by(),
            "game_mode": game_config is not None,
            "style_evaluation": evaluate_style,
            "full_glossary_application": (
                (game_config is None and previous_result is None)
                or (game_config is not None and row_positions is None)
            ),
            "estimated_values": estimate.unique_values,
            "estimated_batches": estimate.batches,
            "estimated_input_tokens": estimate.input_tokens,
            "estimated_output_tokens": estimate.output_tokens,
            "estimated_cost_usd": estimate.estimated_cost_usd,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
        }
        with self.lock:
            self.jobs[job_id] = {
                "future": None, "done": 0, "total": estimate.batches, "label": "Starting",
                "cancel_event": cancel_event, "metadata": metadata,
            }

        def update(done: int, total: int, label: str) -> None:
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id].update(done=done, total=total, label=label)

        def run():
            if demo_mode:
                backend = DemoTranslationBackend(
                    delay_seconds=float(os.getenv("DEMO_TRANSLATION_DELAY_SECONDS", "0.5")),
                    # A normal demo run keeps failing marked values so fallback can isolate them.
                    # A targeted retry disables the artificial failure and succeeds automatically.
                    failure_marker=None if previous_result is not None or game_retry_failures else "[FAIL]",
                )
            else:
                backend = OpenAITranslationBackend(
                    OpenAI(
                        api_key=key,
                        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
                        max_retries=0,
                    ),
                    model=model,
                )
            escalation_model = None if demo_mode else os.getenv("OPENAI_ESCALATION_MODEL") or None
            agent = TranslationAgent(
                backend,
                batch_size=batch_size,
                max_retries=int(os.getenv("TRANSLATION_MAX_RETRIES", "1")),
                escalation_model=escalation_model,
            )
            try:
                if game_config is not None:
                    game_agent = GameLocalizationAgent(
                        backend,
                        batch_size=batch_size,
                        max_retries=int(os.getenv("TRANSLATION_MAX_RETRIES", "1")),
                        escalation_model=escalation_model,
                    )
                    return game_agent.run(
                        dataframe,
                        game_config,
                        character_bible if character_bible is not None else default_character_bible(dataframe, game_config),
                        languages,
                        glossary=glossary,
                        progress=update,
                        should_cancel=cancel_event.is_set,
                        row_positions=row_positions,
                        base_result=game_base_result,
                        approved_locks=game_approved_locks,
                        evaluate_style=evaluate_style,
                        evaluate_hallucination=hallucination_check_enabled(),
                        hallucination_sample_rate=float(os.getenv("TRANSLATION_HALLUCINATION_SAMPLE_RATE", "0.2")),
                        hallucination_similarity_threshold=float(
                            os.getenv("TRANSLATION_HALLUCINATION_THRESHOLD", "0.45")
                        ),
                        context_overrides=context_overrides,
                    )
                if previous_result is not None:
                    return agent.retry_failures(
                        dataframe, previous_result, progress=update, glossary=glossary,
                        should_cancel=cancel_event.is_set,
                    )
                return agent.run(
                    dataframe, source_columns, languages, progress=update, glossary=glossary,
                    should_cancel=cancel_event.is_set,
                )
            finally:
                with self.lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]["metadata"]["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        future = self.executor.submit(run)
        with self.lock:
            self.jobs[job_id]["future"] = future
        return job_id

    def status(self, job_id: str | None) -> dict:
        if not job_id:
            return {"state": "none"}
        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return {"state": "missing"}
            future = record["future"]
            return {
                "state": (
                    "cancelled" if future is not None and future.cancelled()
                    else "completed" if future is not None and future.done()
                    else "cancelling" if record["cancel_event"].is_set()
                    else "running"
                ),
                "done": record["done"],
                "total": record["total"],
                "label": record["label"],
                "metadata": dict(record["metadata"]),
            }

    def cancel(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        with self.lock:
            record = self.jobs.get(job_id)
            if not record:
                return False
            record["cancel_event"].set()
            record["label"] = "Cancelling safely"
            future = record["future"]
            if future is not None:
                future.cancel()
        return True

    def consume(self, job_id: str):
        with self.lock:
            record = self.jobs.pop(job_id)
        return record["future"].result()


@st.cache_resource
def _background_jobs(manager_version: str) -> BackgroundJobManager:
    """Versioned resource so hot reloads cannot retain an incompatible manager."""
    return BackgroundJobManager()


def background_jobs() -> BackgroundJobManager:
    return _background_jobs("string-package-v2")


def start_background_job(document: dict, *args, **kwargs) -> bool:
    """Start work without allowing a transient submit error to disappear on rerun."""
    document["running_input_snapshot"] = {
        "bible_revision": document.get("bible_revision", 0),
        "glossary_revision": document.get("glossary_revision", 0),
        "target_languages": list(document.get("target_languages", [])),
    }
    try:
        document["job_id"] = background_jobs().submit(*args, **kwargs)
    except Exception as error:
        message = f"Translation could not start: {error}"
        document["job_id"] = None
        document.pop("running_input_snapshot", None)
        document["last_job_error"] = message
        add_message(document, "assistant", message, notify=True)
        return False
    document["last_job_error"] = ""
    return True


def api_key() -> str | None:
    if demo_mode_enabled():
        return "local-demo-no-api-key"
    try:
        secret = st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        secret = None
    return secret or os.getenv("OPENAI_API_KEY")


def read_csv(uploaded_file) -> pd.DataFrame:
    """Try common encodings while keeping every input column intact."""
    raw = uploaded_file.getvalue()
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "big5", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError("Could not decode this CSV as UTF-8, Big5, or GB18030. " + "; ".join(errors))


def initial_messages() -> list[dict[str, str]]:
    return [{
        "role": "assistant",
        "content": (
            "Use the guided actions to prepare the Character Bible, target languages, and optional terminology. "
            "The fixed message composer saves optional instructions; Translate starts the localization job."
        ),
    }]


def add_message(document: dict, role: str, content: str, notify: bool = False) -> None:
    document["messages"].append({"role": role, "content": content})
    if role == "assistant":
        document["unread"] = notify or document["id"] != st.session_state.get("active_document_id")


def ensure_no_failed_values_message(document: dict) -> None:
    result = document.get("result")
    if result is None or not result.failures.empty:
        return
    notice = "✓ No failed values to retry."
    if not any(message.get("content") == notice for message in document["messages"]):
        add_message(document, "assistant", notice)


def keep_conversation_scrolled_to_latest() -> None:
    """Fit Conversation above its composer and keep the newest message visible."""
    st.components.v1.html(
        """
        <script>
        (() => {
          let doc;
          try {
            doc = window.parent.document;
          } catch (_) {
            return;
          }
          const findComposer = () => {
            const bottom = doc.querySelector('[data-testid="stBottomBlockContainer"]');
            if (!bottom) return null;
            return bottom.querySelector(
              '[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child'
            ) || bottom.querySelector('[data-testid="stColumn"]');
          };
          const fit = () => {
            const mainScroller = doc.querySelector(
              '[data-testid="stAppScrollToBottomContainer"]'
            );
            if (mainScroller && mainScroller.scrollTop !== 0) {
              mainScroller.scrollTop = 0;
            }
            const history = doc.querySelector('.st-key-conversation_history');
            const resultPanel = doc.querySelector('.st-key-persistent_result_panel');
            const composer = findComposer();
            if (!history || !composer) return;
            const messageCount = history.querySelectorAll(
              '[data-testid="stChatMessage"]'
            ).length;
            const previousMessageCount = Number(
              history.dataset.messageCount || -1
            );
            const shouldScrollToLatest = previousMessageCount < messageCount;
            history.dataset.messageCount = String(messageCount);
            const top = history.getBoundingClientRect().top;
            const bottom = composer.getBoundingClientRect().top;
            const height = Math.max(80, bottom - top);
            doc.documentElement.style.setProperty(
              '--conversation-available-height', `${height}px`
            );
            const historyWrapper = history.parentElement?.closest(
              '[data-testid="stLayoutWrapper"]'
            );
            if (historyWrapper) {
              historyWrapper.style.height = `${height}px`;
              historyWrapper.style.minHeight = '0px';
              historyWrapper.style.maxHeight = `${height}px`;
              historyWrapper.style.overflow = 'hidden';
            }
            if (resultPanel) {
              const resultTop = resultPanel.getBoundingClientRect().top;
              const resultHeight = Math.max(
                160, doc.defaultView.innerHeight - resultTop
              );
              resultPanel.style.height = `${resultHeight}px`;
              resultPanel.style.minHeight = `${resultHeight}px`;
              resultPanel.style.maxHeight = `${resultHeight}px`;
            }
            history.style.height = `${height}px`;
            history.style.maxHeight = `${height}px`;
            history.style.overflowY = 'auto';

            // Messages start at the top. Once their total height exceeds the
            // viewport, the scroll-to-latest behavior below pushes older
            // messages upward like a conventional chat.
            const content = history.matches('[data-testid="stVerticalBlock"]')
              ? history
              : Array.from(
                  history.querySelectorAll('[data-testid="stVerticalBlock"]')
                ).find((element) => element.closest('.st-key-conversation_history') === history);
            if (content) {
              content.style.minHeight = content === history ? '0px' : '100%';
              content.style.display = 'flex';
              content.style.flexDirection = 'column';
              const firstItem = content.firstElementChild;
              if (firstItem) firstItem.style.marginTop = '0px';
            }
            if (shouldScrollToLatest) {
              history.scrollTop = history.scrollHeight;
            }
          };
          requestAnimationFrame(() => requestAnimationFrame(fit));
          setTimeout(fit, 100);
          setTimeout(fit, 350);
          const observer = new ResizeObserver(fit);
          observer.observe(doc.documentElement);
          const composer = findComposer();
          if (composer) observer.observe(composer);
          const history = doc.querySelector('.st-key-conversation_history');
          if (history) {
            observer.observe(history);
            const mutations = new MutationObserver(() => requestAnimationFrame(fit));
            mutations.observe(history, {childList: true, subtree: true});
          }
          window.addEventListener('resize', fit);
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def save_active_draft() -> None:
    active_id = st.session_state.get("active_document_id")
    document = st.session_state.get("documents", {}).get(active_id)
    if not document:
        return
    draft_key = f"draft_input_{active_id}"
    if draft_key in st.session_state:
        document["draft"] = st.session_state[draft_key]


def switch_document(document_id: str) -> None:
    save_active_draft()
    active_id = st.session_state.get("active_document_id")
    active_document = st.session_state.get("documents", {}).get(active_id)
    if active_document and active_document.get("conversation_action") == "rerun":
        abandon_conversation_rerun(active_id)
    if active_document and active_document.get("conversation_action") == "languages":
        complete_conversation_language_action(active_id)
    if active_document and active_document.get("conversation_action") == "glossary":
        active_document["conversation_action"] = ""
        add_message(active_document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")
    st.session_state.active_document_id = document_id
    document = st.session_state.documents[document_id]
    document["unread"] = False
    st.session_state[f"draft_input_{document_id}"] = document.get("draft", "")


def save_target_languages(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    document["target_languages"] = list(
        st.session_state.get(f"target_languages_{document_id}", [])
    )


def composed_translation_prompt(document_id: str) -> str:
    """Combine optional chat instructions with the language multi-select."""
    document = st.session_state.documents[document_id]
    draft = str(document.get("translation_instructions") or "").strip()
    # The callback persists the widget value on the document so it survives project switches
    # and remains available while the results column renders before the composer widget.
    selected = list(document.get("target_languages", []))
    if not selected:
        return draft
    language_instruction = "Translate to " + " and ".join(selected)
    if not draft:
        return language_instruction
    if all(language in parse_languages(draft) for language in selected):
        return draft
    return f"{draft}\n\nTarget languages: {', '.join(selected)}"


def send_optional_instruction(document_id: str, instruction: str | None = None) -> None:
    """Send a persistent chat message without starting a translation job."""
    document = st.session_state.documents[document_id]
    instruction = str(
        instruction if instruction is not None
        else st.session_state.get(f"draft_input_{document_id}") or ""
    ).strip()
    if not instruction:
        document["empty_prompt_error"] = True
        return
    document["empty_prompt_error"] = False
    document["translation_instructions"] = instruction
    add_message(document, "user", instruction)
    add_message(
        document,
        "assistant",
        "Optional instructions saved. I’ll use them for the next translation.",
    )
    document["draft"] = ""


def queue_message(document_id: str, allow_missing_bible: bool = False) -> None:
    document = st.session_state.documents[document_id]
    draft_key = f"draft_input_{document_id}"
    prompt = composed_translation_prompt(document_id)
    if not prompt:
        document["empty_prompt_error"] = True
        document["awaiting_bible_override"] = False
        return
    document["empty_prompt_error"] = False
    missing_guidance = missing_bible_speakers(document)
    if missing_guidance and not allow_missing_bible:
        document["send_bible_warning"] = (
            f"Character Bible is incomplete for {len(missing_guidance)} speaker(s): "
            f"{', '.join(missing_guidance)}. Translation has not started, so no API tokens were used. "
            "Complete the Character Bible, or explicitly continue with a neutral voice."
        )
        document["awaiting_bible_override"] = True
        return
    document["send_bible_warning"] = ""
    document["awaiting_bible_override"] = False
    document["pending_prompt"] = prompt


def mark_bible_changed(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    document["bible_revision"] = document.get("bible_revision", 0) + 1


def show_chat_setup(document_id: str, panel: str) -> None:
    """Open one guided setup action directly inside Conversation."""
    st.session_state.documents[document_id]["chat_setup_panel"] = panel


def render_chat_character_bible_upload(document_id: str) -> None:
    """Validate and import a Character Bible without sending the user to the workspace."""
    document = st.session_state.documents[document_id]
    bible_upload = st.file_uploader(
        "Upload Character Bible CSV",
        type=["csv"],
        key=f"chat_bible_upload_{document_id}",
    )
    if bible_upload is None:
        st.caption("The CSV must contain a `speaker` column and character-guidance fields.")
        return
    bible_hash = hashlib.sha256(bible_upload.getvalue()).hexdigest()
    if bible_hash == document.get("bible_upload_hash"):
        st.success("Character Bible already imported.")
        return
    try:
        imported = read_csv(bible_upload)
    except Exception as error:
        st.error(f"I couldn’t read this Character Bible CSV. Details: {error}")
        return
    bible_errors = validate_character_bible(imported)
    if bible_errors:
        st.error(
            "This file does not look like a Character Bible. The current Character Bible was not replaced.\n\n- "
            + "\n- ".join(bible_errors)
        )
        return
    current = document.get("character_bible")
    if current is None or "speaker" not in current.columns:
        current = default_character_bible(
            document["dataframe"], GameConfig(**document["game_config"])
        )
    for column in current.columns:
        if column not in imported:
            imported[column] = ""
    document["character_bible"] = imported[list(current.columns)].fillna("")
    document["character_bible_name"] = bible_upload.name
    document["bible_upload_hash"] = bible_hash
    document["bible_revision"] = document.get("bible_revision", 0) + 1
    document["character_bible_editor_version"] = (
        document.get("character_bible_editor_version", 0) + 1
    )
    touch_workspace_item(document, "bible")
    document["send_bible_warning"] = ""
    document["awaiting_bible_override"] = False
    add_message(document, "user", f"Uploaded Character Bible: **{bible_upload.name}**")
    add_message(
        document,
        "assistant",
        "Character Bible validated and saved. Character guidance is ready for translation.",
    )
    ensure_game_status_message(document)
    st.success("Character Bible imported. Character guidance is ready for translation.")


@st.dialog("Character Bible", width="large")
def pending_character_bible_drawer() -> None:
    """Let users inspect and edit a Bible before translation data is uploaded."""
    bible = st.session_state.get("pending_character_bible")
    if bible is None or bible.empty:
        st.info("Upload a Character Bible from Conversation first.")
        return
    st.caption("This Character Bible will be attached to the next translation-data file you upload.")
    _, save_column, cancel_column = st.columns([6, 1.35, 0.8])
    with save_column:
        save_bible = st.button(
            "Save Character Bible",
            type="primary",
            width="content",
            key="save_pending_bible",
        )
    with cancel_column:
        cancel_bible = st.button(
            "Cancel",
            width="content",
            key="close_pending_bible",
        )
    edited_bible = st.data_editor(
        bible,
        key=f"pending_character_bible_editor_{st.session_state.get('pending_bible_editor_version', 0)}",
        width="stretch",
        height=520,
        hide_index=True,
        num_rows="dynamic",
        disabled=False,
    )
    if save_bible:
        edited_bible = edited_bible.fillna("")
        nonempty_rows = edited_bible.astype(str).apply(
            lambda column: column.str.strip()
        ).ne("").any(axis=1)
        edited_bible = edited_bible.loc[nonempty_rows].reset_index(drop=True)
        errors = validate_character_bible(edited_bible)
        if errors:
            st.error("Fix these Character Bible issues before saving:\n\n- " + "\n- ".join(errors))
        else:
            st.session_state.pending_character_bible = edited_bible
            st.session_state.pending_bible_editor_version = (
                st.session_state.get("pending_bible_editor_version", 0) + 1
            )
            st.session_state.onboarding_messages = [
                *st.session_state.get("onboarding_messages", []),
                {"role": "assistant", "content": "Character Bible changes saved."},
            ]
            st.success("Character Bible changes saved.")
    if cancel_bible:
        st.session_state.pending_bible_editor_version = (
            st.session_state.get("pending_bible_editor_version", 0) + 1
        )
        st.rerun()


def render_pending_character_bible_summary() -> None:
    """Show the staged Character Bible instead of a blank onboarding workspace."""
    bible = st.session_state.get("pending_character_bible")
    if bible is None or bible.empty or "speaker" not in bible.columns:
        return
    normalized = bible.fillna("")
    guidance_columns = [column for column in normalized.columns if column != "speaker"]
    speaker_rows = normalized["speaker"].astype(str).str.strip().ne("")
    guided_rows = (
        normalized.loc[speaker_rows, guidance_columns]
        .astype(str)
        .apply(lambda column: column.str.strip())
        .ne("")
        .any(axis=1)
        if guidance_columns else pd.Series(False, index=normalized.index)
    )
    with st.container(border=True, key="pending_character_bible_summary"):
        title_column, speakers_column, guidance_column = st.columns([2, 1, 1])
        title_column.markdown(
            f"**Character Bible**  \n{st.session_state.get('pending_character_bible_name') or 'Character Bible'}"
        )
        speakers_column.metric("Speakers", int(speaker_rows.sum()))
        guidance_column.metric(
            "Guidance ready",
            f"{int(guided_rows.sum())}/{int(speaker_rows.sum())}",
        )
        if st.button("Open Character Bible", key="open_pending_character_bible", width="stretch"):
            pending_character_bible_drawer()


@st.dialog("Glossary", width="large")
def pending_glossary_drawer() -> None:
    """Edit glossary rules before a translation-data file exists."""
    editor_key = "pending_glossary_editor"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = st.session_state.get("onboarding_glossary_text", "")
    st.caption(
        "Use source | language | translation, optionally followed by type | notes. "
        "A single term means do not translate."
    )
    upload_column, _, apply_column, cancel_column = st.columns([1.45, 5, 1.1, 0.8])
    upload_key = f"pending_drawer_glossary_upload_{st.session_state.uploader_version}"
    with upload_column, st.container(key="glossary_upload_compact"):
        st.file_uploader(
            "Upload",
            type=["csv"],
            key=upload_key,
            label_visibility="collapsed",
            on_change=handle_onboarding_glossary_upload,
            args=(upload_key,),
        )
    with apply_column:
        apply_rules = st.button("Apply glossary", type="primary", width="content", key="apply_pending_glossary")
    with cancel_column:
        cancel = st.button("Cancel", width="content", key="close_pending_glossary")
    draft = st.text_area(
        "Terminology rules",
        key=editor_key,
        height=420,
        placeholder="OpenAI\n會員 | English | member",
    )
    if apply_rules:
        _, errors = parse_glossary(draft)
        if errors:
            st.error("\n".join(errors))
        else:
            apply_onboarding_chat_glossary(draft)
            st.success("Glossary applied. It will be attached to the next translation file.")
    if cancel:
        st.rerun()


def render_pending_glossary_summary() -> None:
    """Keep the independent glossary entry point visible before file upload."""
    glossary_text = str(st.session_state.get("onboarding_glossary_text", ""))
    if not glossary_text.strip():
        return
    entries, _ = parse_glossary(glossary_text)
    languages = {entry.target_language for entry in entries if entry.target_language}
    with st.container(border=True, key="pending_glossary_summary"):
        title_column, rules_column, language_column = st.columns([2, 1, 1])
        title_column.markdown("**Terminology rules**  \nGlossary & protected terms")
        rules_column.metric("Rules", len(entries))
        language_column.metric("Languages", len(languages))
        if st.button("Open Glossary", key="open_pending_glossary", width="stretch"):
            pending_glossary_drawer()


@st.dialog("Character Bible", width="large")
def character_bible_drawer(document_id: str) -> None:
    """Edit character guidance in a right-side drawer instead of the main workspace."""
    document = st.session_state.documents[document_id]
    dataframe = document["dataframe"]
    current_config = GameConfig(**document["game_config"]) if document.get("game_config") else GameConfig()
    st.caption("Add one row per speaker. These fields guide voice, tone, pronouns, and relationships.")
    if document.get("character_bible") is None or "speaker" not in document["character_bible"].columns:
        document["character_bible"] = default_character_bible(dataframe, current_config)

    st.markdown(
        """
        <style>
        .st-key-bible_upload_compact [data-testid="stFileUploaderDropzone"] {
            min-height: 2.5rem;
            padding: 0;
            border: 0;
            background: transparent;
        }
        .st-key-bible_upload_compact [data-testid="stFileUploaderDropzoneInstructions"],
        .st-key-bible_upload_compact small {
            display: none;
        }
        .st-key-bible_upload_compact [data-testid="stBaseButton-secondary"] {
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    upload_column, _, save_column, cancel_column = st.columns([1.45, 5, 1.35, 0.8])
    with upload_column, st.container(key="bible_upload_compact"):
        bible_upload = st.file_uploader(
            "Upload",
            type=["csv"],
            key=f"bible_upload_{document_id}",
            label_visibility="collapsed",
        )
    with save_column:
        save_bible = st.button(
            "Save Character Bible",
            type="primary",
            width="content",
            key=f"save_bible_{document_id}",
        )
    with cancel_column:
        cancel_bible = st.button(
            "Cancel",
            width="content",
            key=f"close_bible_{document_id}",
        )

    if bible_upload is not None:
        bible_hash = hashlib.sha256(bible_upload.getvalue()).hexdigest()
        if bible_hash != document.get("bible_upload_hash"):
            try:
                imported = read_csv(bible_upload)
            except Exception as error:
                st.error(
                    "I couldn’t read this Character Bible CSV. "
                    f"Your current Character Bible was not replaced. Details: {error}"
                )
            else:
                bible_errors = validate_character_bible(imported)
                if bible_errors:
                    st.error(
                        "This file does not look like a Character Bible. "
                        "Your current Character Bible was not replaced.\n\n- "
                        + "\n- ".join(bible_errors)
                    )
                else:
                    for column in document["character_bible"].columns:
                        if column not in imported:
                            imported[column] = ""
                    document["character_bible"] = imported[
                        list(document["character_bible"].columns)
                    ].fillna("")
                    document["character_bible_name"] = bible_upload.name
                    document["bible_upload_hash"] = bible_hash
                    document["bible_revision"] = document.get("bible_revision", 0) + 1
                    document["character_bible_editor_version"] = (
                        document.get("character_bible_editor_version", 0) + 1
                    )
                    touch_workspace_item(document, "bible")

    st.info("Edit any cell below. You can also add or delete speaker rows, then save your changes.")
    edited_bible = st.data_editor(
        document["character_bible"],
        key=(
            f"character_bible_{document_id}_"
            f"{document.get('character_bible_editor_version', 0)}"
        ),
        width="stretch",
        height=520,
        hide_index=True,
        num_rows="dynamic",
        disabled=False,
    )

    if save_bible:
        edited_bible = edited_bible.fillna("")
        nonempty_rows = edited_bible.astype(str).apply(
            lambda column: column.str.strip()
        ).ne("").any(axis=1)
        edited_bible = edited_bible.loc[nonempty_rows].reset_index(drop=True)
        bible_errors = validate_character_bible(edited_bible)
        if bible_errors:
            st.error("Fix these Character Bible issues before saving:\n\n- " + "\n- ".join(bible_errors))
        else:
            current_bible = document["character_bible"].fillna("").reset_index(drop=True)
            if edited_bible.equals(current_bible):
                st.info("No Character Bible changes to save.")
            else:
                document["character_bible"] = edited_bible
                document["bible_revision"] = document.get("bible_revision", 0) + 1
                document["character_bible_editor_version"] = (
                    document.get("character_bible_editor_version", 0) + 1
                )
                touch_workspace_item(document, "bible")
                document["send_bible_warning"] = ""
                document["awaiting_bible_override"] = False
                add_message(document, "assistant", "Character Bible changes saved.")
                ensure_game_status_message(document)
                st.success("Character Bible changes saved. Updated guidance is ready to use.")

    if cancel_bible:
        document["character_bible_editor_version"] = (
            document.get("character_bible_editor_version", 0) + 1
        )
        st.rerun()


def render_character_bible_summary(document_id: str) -> None:
    """Keep an imported Character Bible visible from the persistent workspace."""
    document = st.session_state.documents[document_id]
    bible = document.get("character_bible")
    if (
        not document.get("game_mode")
        or document.get("bible_revision", 0) <= 0
        or bible is None
        or bible.empty
        or "speaker" not in bible.columns
    ):
        return

    normalized = bible.fillna("")
    guidance_columns = [column for column in normalized.columns if column != "speaker"]
    speaker_rows = normalized["speaker"].astype(str).str.strip().ne("")
    guided_rows = normalized.loc[speaker_rows, guidance_columns].astype(str).apply(
        lambda column: column.str.strip()
    ).ne("").any(axis=1) if guidance_columns else pd.Series(False, index=normalized.index)
    speaker_count = int(speaker_rows.sum())
    guided_count = int(guided_rows.sum())

    with st.container(border=True, key=f"character_bible_summary_{document_id}"):
        title_column, speakers_column, guidance_column = st.columns([2, 1, 1])
        bible_name = document.get("character_bible_name") or "Character Bible"
        title_column.markdown(f"**Character Bible**  \n{bible_name}")
        speakers_column.metric("Speakers", speaker_count)
        guidance_column.metric("Guidance ready", f"{guided_count}/{speaker_count}")
        if st.button(
            "Open Character Bible",
            key=f"open_character_bible_{document_id}",
            width="stretch",
        ):
            character_bible_drawer(document_id)


def render_active_file_summary(document_id: str) -> None:
    """Render the translation-data card and its setup entry point."""
    document = st.session_state.documents[document_id]
    dataframe = document["dataframe"]
    with st.container(border=True, key=f"active_file_summary_{document_id}"):
        name_column, rows_column, columns_column = st.columns([2, 1, 1])
        name_column.markdown(f"**Active file**  \n{document['name']}")
        rows_column.metric("Rows", f"{len(dataframe):,}")
        columns_column.metric("Columns", len(dataframe.columns))
        if st.button(
            "Open Translation Setup",
            key=f"open_translation_setup_{document_id}",
            width="stretch",
        ):
            translation_setup_drawer(document_id)


@st.dialog("Glossary", width="large")
def glossary_drawer(document_id: str) -> None:
    """Edit, validate, and apply project terminology on a dedicated page."""
    document = st.session_state.documents[document_id]
    glossary_key = f"glossary_{document_id}"
    if glossary_key not in st.session_state:
        st.session_state[glossary_key] = document.get("glossary_text", "")
    st.caption(
        "Use source | language | translation, optionally followed by type | notes. "
        "A single term means do not translate."
    )
    upload_column, _, apply_column, cancel_column = st.columns([1.45, 5, 1.1, 0.8])
    upload_key = f"drawer_glossary_upload_{document_id}_{document.get('glossary_revision', 0)}"
    with upload_column, st.container(key="glossary_upload_compact"):
        st.file_uploader(
            "Upload",
            type=["csv"],
            key=upload_key,
            label_visibility="collapsed",
            on_change=handle_project_glossary_upload,
            args=(document_id, upload_key),
        )
    with apply_column:
        st.button(
            "Apply glossary",
            type="primary",
            width="content",
            key=f"drawer_apply_glossary_{document_id}",
            on_click=apply_glossary,
            args=(document_id,),
        )
    with cancel_column:
        cancel = st.button("Cancel", width="content", key=f"close_glossary_{document_id}")
    glossary_draft = st.text_area(
        "Terminology rules",
        key=glossary_key,
        height=420,
        placeholder="OpenAI\n會員 | English | member",
        on_change=stage_glossary_draft,
        args=(document_id,),
    )
    document["glossary_draft"] = glossary_draft
    if document.get("glossary_validation_errors"):
        st.error("\n".join(document["glossary_validation_errors"]))
    elif document.get("glossary_confirmation"):
        st.success(document["glossary_confirmation"])
    if cancel:
        st.rerun()


def render_glossary_summary(document_id: str) -> None:
    """Render the persistent glossary card in the project workspace."""
    document = st.session_state.documents[document_id]
    entries, _ = parse_glossary(document.get("glossary_text", ""))
    with st.container(border=True, key=f"glossary_summary_{document_id}"):
        title_column, rules_column, language_column = st.columns([2, 1, 1])
        title_column.markdown(f"**Terminology rules**  \nGlossary version {document.get('glossary_revision', 0)}")
        rules_column.metric("Rules", len(entries))
        languages = {entry.target_language for entry in entries if entry.target_language}
        language_column.metric("Languages", len(languages))
        if st.button(
            "Open Glossary",
            key=f"open_glossary_{document_id}",
            width="stretch",
        ):
            glossary_drawer(document_id)


@st.dialog("Translation setup", width="large")
def translation_setup_drawer(document_id: str) -> None:
    """Render translation and game-context configuration on a dedicated page."""
    document = st.session_state.documents[document_id]
    dataframe = document["dataframe"]
    profiles = document["profiles"]
    proposed = [profile.name for profile in profiles if profile.selected]

    document["game_mode"] = st.checkbox(
        "Game string-package localization mode",
        value=document.get("game_mode", False),
        key=f"game_mode_{document_id}",
        help="Adds speaker personality, scene context, review workflow, game QA, and targeted reruns.",
    )
    if document.get("game_mode"):
        current_config = (
            GameConfig(**document["game_config"])
            if document.get("game_config") else GameConfig()
        )
        if current_config.speaker:
            missing_guidance = missing_bible_speakers(document)
            if missing_guidance:
                st.caption(
                    f"Character Bible · {len(missing_guidance)} speaker(s) still need guidance. "
                    "Upload it from Conversation."
                )
            else:
                st.caption("Character Bible · ready")
        else:
            st.caption("Character Bible · not required because no speaker column was detected")

        column_options = list(dataframe.columns)
        required_left, required_right = st.columns(2)
        speaker_options = ["(none)"] + column_options
        speaker_default = (
            speaker_options.index(current_config.speaker)
            if current_config.speaker in speaker_options else 0
        )
        source_default = (
            column_options.index(current_config.source_text)
            if current_config.source_text in column_options
            else next((column_options.index(name) for name in proposed if name in column_options), 0)
        )
        speaker_column = required_left.selectbox(
            "Speaker column",
            speaker_options,
            index=speaker_default,
            key=f"game_speaker_column_{document_id}",
        )
        speaker_column = "" if speaker_column == "(none)" else speaker_column
        source_column = required_right.selectbox(
            "Dialogue column",
            column_options,
            index=source_default,
            key=f"game_source_column_{document_id}",
        )

        def optional_game_column(label: str, field: str):
            options = ["(none)"] + column_options
            current = getattr(current_config, field)
            index = options.index(current) if current in options else 0
            selected = st.selectbox(
                label,
                options,
                index=index,
                key=f"game_{field}_column_{document_id}",
            )
            return "" if selected == "(none)" else selected

        with st.expander("Advanced column mapping", expanded=False):
            st.caption("Optional fields used for context, tracking, and UI-length validation.")
            line_id_column = optional_game_column("Line ID", "line_id")
            scene_column = optional_game_column("Scene", "scene_id")
            listener_column = optional_game_column("Listener", "listener")
            emotion_column = optional_game_column("Emotion", "emotion")
            context_column = optional_game_column("Scene context", "context")
            limit_column = optional_game_column("Character limit", "character_limit")
        game_config = GameConfig(
            line_id=line_id_column,
            scene_id=scene_column,
            speaker=speaker_column,
            source_text=source_column,
            listener=listener_column,
            emotion=emotion_column,
            context=context_column,
            character_limit=limit_column,
            previous_lines=current_config.previous_lines,
            next_lines=current_config.next_lines,
        )
        document["selected_columns"] = [game_config.source_text]
        st.caption(
            "Schema: line_id · scene_id · speaker · listener · emotion · source_text · "
            "context · character_limit. Optional columns are used when present."
        )
        with st.expander("Context & style evaluation", expanded=False):
            st.caption(
                "Control neighboring dialogue context and the optional post-translation style review."
            )
            context_left, context_right = st.columns(2)
            previous_lines = context_left.number_input(
                "Previous lines",
                min_value=0,
                max_value=5,
                value=game_config.previous_lines,
                key=f"game_previous_{document_id}",
            )
            next_lines = context_right.number_input(
                "Next lines",
                min_value=0,
                max_value=5,
                value=game_config.next_lines,
                key=f"game_next_{document_id}",
            )
            document["evaluate_style"] = st.checkbox(
                "Run AI style evaluation after translation",
                value=document.get("evaluate_style", False),
                key=f"evaluate_style_{document_id}",
                help="Adds a second model pass with a 0–100 advisory style score.",
            )
        game_config = GameConfig(
            line_id=line_id_column,
            scene_id=scene_column,
            speaker=speaker_column,
            source_text=source_column,
            listener=listener_column,
            emotion=emotion_column,
            context=context_column,
            character_limit=limit_column,
            previous_lines=int(previous_lines),
            next_lines=int(next_lines),
        )
        document["game_config"] = asdict(game_config)

        st.markdown(
            "**Scene & speaker work units**",
            help=(
                "Type a target-language request in Conversation (for example, “Translate to English and Japanese”) "
                "then click Preview workload & cost before sending."
            ),
        )
        units = game_work_units(dataframe, game_config)
        unit_left, unit_middle, unit_right = st.columns(3)
        unit_left.metric("Scenes", units["Scene"].nunique())
        unit_middle.metric("Speakers", units["Speaker"].nunique())
        unit_right.metric("Work units", len(units))
        st.dataframe(
            units,
            hide_index=True,
            width="stretch",
            height=min(420, 36 + len(units) * 35),
        )
        st.caption(
            "Translation batches follow these scene + speaker units. Original row order and neighboring-line context are preserved."
        )
    else:
        selected_columns = st.multiselect(
            "Columns to translate",
            options=list(dataframe.columns),
            default=document.get("selected_columns") or proposed,
            key=f"selected_columns_{document_id}",
            help="This selection is saved independently for this file.",
        )
        document["selected_columns"] = list(selected_columns)
        profile_table = pd.DataFrame({
            "column": [profile.name for profile in profiles],
            "Chinese ratio": [f"{profile.chinese_ratio:.0%}" for profile in profiles],
            "decision": ["suggest" if profile.selected else "skip" for profile in profiles],
            "reason": [profile.reason for profile in profiles],
        })
        st.dataframe(profile_table, hide_index=True, width="stretch")

    if st.button("Close", key=f"close_translation_setup_{document_id}"):
        st.rerun()


def preview_translation_plan(document_id: str) -> None:
    """Validate either workflow and report its pre-run estimate in Conversation."""
    document = st.session_state.documents[document_id]
    abandon_conversation_rerun(document_id)
    if document.get("conversation_action") == "languages":
        complete_conversation_language_action(document_id)
    if document.get("conversation_action") == "glossary":
        document["conversation_action"] = ""
        add_message(document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")
    draft_key = f"draft_input_{document_id}"
    document["draft"] = str(st.session_state.get(draft_key) or "")
    prompt = composed_translation_prompt(document_id)
    languages = parse_languages(prompt)
    if not languages:
        document["plan_preview"] = None
        document["plan_preview_error"] = (
            "Add at least one supported target language—for example, “Translate to English and Japanese.”"
        )
        add_message(document, "assistant", document["plan_preview_error"])
        return
    if document.get("game_mode"):
        if not document.get("game_config"):
            document["plan_preview"] = None
            document["plan_preview_error"] = "Finish the game column mapping before previewing the workload."
            add_message(document, "assistant", document["plan_preview_error"])
            return
        estimate = estimate_game_workload(
            document["dataframe"],
            GameConfig(**document["game_config"]),
            languages,
            document.get("evaluate_style", False),
        )
    else:
        proposed = [profile.name for profile in document["profiles"] if profile.selected]
        selected_columns = list(document.get("selected_columns") or proposed)
        if not selected_columns:
            document["plan_preview"] = None
            document["plan_preview_error"] = (
                "Choose at least one source column in Open Translation Setup before previewing the workload."
            )
            add_message(document, "assistant", document["plan_preview_error"])
            return
        batch_size = int(os.getenv("TRANSLATION_BATCH_SIZE", "25"))
        estimate = TranslationAgent(object(), batch_size=batch_size).estimate(
            document["dataframe"], selected_columns, languages,
            float(os.getenv("OPENAI_INPUT_COST_PER_MILLION", "0.15")),
            float(os.getenv("OPENAI_OUTPUT_COST_PER_MILLION", "0.60")),
        )
    document["plan_preview"] = {
        "prompt": prompt,
        "languages": languages,
        "translations": estimate.unique_values,
        "batches": estimate.batches,
        "tokens": estimate.input_tokens + estimate.output_tokens,
        "cost": estimate.estimated_cost_usd,
    }
    document["plan_preview_error"] = ""
    preview = document["plan_preview"]
    add_message(
        document,
        "assistant",
        f"Estimate ready: **{preview['translations']:,} translations** · "
        f"**{preview['batches']:,} batches** · ~{preview['tokens']:,} tokens · "
        f"~${preview['cost']:.5f} USD. This has not started translation.",
    )


def cancel_translation(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    if background_jobs().cancel(document.get("job_id")):
        add_message(document, "assistant", "Cancellation requested. I’ll stop safely after the current API call finishes.")


def queue_failed_retry(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    document["pending_failed_retry"] = True


def queue_full_rerun(document_id: str) -> None:
    st.session_state.documents[document_id]["pending_full_rerun"] = True


def queue_game_rerun(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    document["pending_game_rerun"] = {
        "scope": st.session_state.get(f"game_rerun_scope_{document_id}", "Selected lines"),
        "value": st.session_state.get(f"game_rerun_value_{document_id}", ""),
    }


def complete_conversation_language_action(document_id: str) -> None:
    """Persist a language picker selection before opening the next chat action."""
    document = st.session_state.documents[document_id]
    if document.get("conversation_action") != "languages":
        return
    languages = list(
        st.session_state.get(
            f"conversation_languages_{document_id}",
            document.get("target_languages", []),
        )
    )
    document["target_languages"] = languages
    document["conversation_action"] = ""
    if languages:
        add_message(document, "user", f"Target languages: **{', '.join(languages)}**")
        add_message(document, "assistant", f"Target languages saved: **{', '.join(languages)}**.")


def complete_onboarding_language_action() -> None:
    """Persist the onboarding language picker before opening another action."""
    if st.session_state.get("onboarding_action") != "languages":
        return
    languages = list(
        st.session_state.get(
            "onboarding_conversation_languages",
            st.session_state.get("onboarding_target_languages", []),
        )
    )
    st.session_state.onboarding_target_languages = languages
    st.session_state.onboarding_action = ""
    if languages:
        st.session_state.onboarding_messages = [
            *st.session_state.get("onboarding_messages", []),
            {"role": "user", "content": f"Target languages: **{', '.join(languages)}**"},
            {"role": "assistant", "content": f"Target languages saved: **{', '.join(languages)}**."},
        ]


def abandon_conversation_rerun(document_id: str) -> None:
    """Close an unsubmitted targeted-rerun picker before another action starts."""
    document = st.session_state.documents[document_id]
    if document.get("conversation_action") != "rerun":
        return
    document["conversation_action"] = ""
    add_message(document, "assistant", "Targeted rerun cancelled before it started.")


def open_conversation_action(document_id: str, action: str) -> None:
    """Open a guided choice as the newest item in this project's conversation."""
    document = st.session_state.documents[document_id]
    if document.get("conversation_action") == "rerun" and action != "rerun":
        abandon_conversation_rerun(document_id)
    if document.get("conversation_action") == "languages" and action != "languages":
        complete_conversation_language_action(document_id)
    if document.get("conversation_action") == "glossary" and action != "glossary":
        add_message(document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")
    prompts = {
        "translation_upload": "Please upload the translation-data CSV you want to open.",
        "bible_upload": "Please upload a Character Bible CSV for this project.",
    }
    prompt = prompts.get(action)
    if prompt and (
        not document.get("messages")
        or document["messages"][-1].get("content") != prompt
    ):
        add_message(document, "assistant", prompt)
    document["conversation_action"] = action


def open_onboarding_language_action() -> None:
    open_onboarding_action("languages")


def open_onboarding_action(action: str) -> None:
    if st.session_state.get("onboarding_action") == "languages" and action != "languages":
        complete_onboarding_language_action()
    if st.session_state.get("onboarding_action") == "glossary" and action != "glossary":
        st.session_state.onboarding_messages = [
            *st.session_state.get("onboarding_messages", []),
            {"role": "assistant", "content": "Glossary editing cancelled. No terminology rules were changed."},
        ]
    prompts = {
        "translation_upload": "Please upload the translation-data CSV you want to open.",
        "bible_upload": "Please upload a Character Bible CSV. You can upload this before the translation data.",
    }
    prompt = prompts.get(action)
    messages = st.session_state.get("onboarding_messages", [])
    if prompt and (not messages or messages[-1].get("content") != prompt):
        st.session_state.onboarding_messages = [
            *messages,
            {"role": "assistant", "content": prompt},
        ]
    st.session_state.onboarding_action = action


def cancel_conversation_action(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    action = document.get("conversation_action", "")
    document["conversation_action"] = ""
    if action == "glossary":
        add_message(document, "user", "Cancel glossary editing")
        add_message(document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")


def cancel_onboarding_action() -> None:
    action = st.session_state.get("onboarding_action", "")
    st.session_state.onboarding_action = ""
    if action == "glossary":
        st.session_state.onboarding_messages = [
            *st.session_state.get("onboarding_messages", []),
            {"role": "user", "content": "Cancel glossary editing"},
            {"role": "assistant", "content": "Glossary editing cancelled. No terminology rules were changed."},
        ]


def apply_chat_glossary(document_id: str, glossary_text: str) -> None:
    document = st.session_state.documents[document_id]
    entries, errors = parse_glossary(glossary_text)
    add_message(document, "user", f"Apply glossary:\n\n```text\n{glossary_text}\n```")
    if errors:
        add_message(document, "assistant", "I couldn’t apply that glossary: " + "; ".join(errors))
        return
    document["glossary_text"] = glossary_text
    document["glossary_draft"] = glossary_text
    document["glossary_dirty"] = False
    document["glossary_validation_errors"] = []
    document["glossary_revision"] = document.get("glossary_revision", 0) + 1
    touch_workspace_item(document, "glossary")
    document["glossary_rerun_ready"] = document.get("result") is not None
    document["conversation_action"] = ""
    add_message(
        document,
        "assistant",
        f"Applied **{len(entries)} terminology rule(s)**."
        + (" You can now start a targeted rerun." if document.get("result") is not None else ""),
    )


def apply_onboarding_chat_glossary(glossary_text: str) -> None:
    entries, errors = parse_glossary(glossary_text)
    st.session_state.onboarding_messages = [
        *st.session_state.get("onboarding_messages", []),
        {"role": "user", "content": f"Apply glossary:\n\n```text\n{glossary_text}\n```"},
    ]
    if errors:
        st.session_state.onboarding_messages.append({
            "role": "assistant",
            "content": "I couldn’t apply that glossary: " + "; ".join(errors),
        })
        return
    st.session_state.onboarding_glossary_text = glossary_text
    touch_pending_workspace_item("glossary")
    st.session_state.onboarding_action = ""
    st.session_state.onboarding_messages.append({
        "role": "assistant",
        "content": f"Applied **{len(entries)} terminology rule(s)**. I’ll carry them into the project.",
    })


def apply_conversation_languages(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    languages = list(st.session_state.get(f"conversation_languages_{document_id}", []))
    if not languages:
        return
    document["target_languages"] = languages
    document["conversation_action"] = ""
    add_message(document, "user", ", ".join(languages))
    add_message(document, "assistant", f"Target languages set to **{', '.join(languages)}**.")


def update_conversation_languages(document_id: str) -> None:
    st.session_state.documents[document_id]["target_languages"] = list(
        st.session_state.get(f"conversation_languages_{document_id}", [])
    )


def apply_onboarding_languages() -> None:
    languages = list(st.session_state.get("onboarding_conversation_languages", []))
    if not languages:
        return
    st.session_state.onboarding_target_languages = languages
    st.session_state.onboarding_action = ""
    st.session_state.onboarding_messages = [
        *st.session_state.get("onboarding_messages", []),
        {"role": "user", "content": ", ".join(languages)},
        {"role": "assistant", "content": f"Target languages set to **{', '.join(languages)}**."},
    ]


def update_onboarding_languages() -> None:
    st.session_state.onboarding_target_languages = list(
        st.session_state.get("onboarding_conversation_languages", [])
    )


def apply_conversation_rerun(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    update_key = rerun_update_widget_key(document)
    selected_updates = list(st.session_state.get(update_key, []))
    available_updates = pending_rerun_updates(document)
    selected_updates = [label for label in selected_updates if label in available_updates]
    if selected_updates:
        document["conversation_action"] = ""
        document["pending_rerun_updates"] = selected_updates
        document["pending_full_rerun"] = True
        add_message(
            document,
            "user",
            "Rerun with updated inputs: **" + ", ".join(selected_updates) + "**",
        )


def pending_rerun_updates(document: dict) -> list[str]:
    """Return only inputs that changed after the latest full translation."""
    if document.get("result") is None:
        return []
    updates: list[str] = []
    if document.get("bible_revision", 0) > document.get("result_bible_revision", 0):
        updates.append("Character Bible updated")
    current_languages = set(document.get("target_languages", []))
    result_languages = set(document.get("result_target_languages", []))
    if current_languages != result_languages:
        updates.append("Target languages updated")
    if document.get("glossary_revision", 0) > document.get("result_glossary_revision", 0):
        updates.append("Glossary updated")
    return updates


def rerun_update_widget_key(document: dict) -> str:
    """Version update pills so every newly changed input is preselected."""
    language_signature = hashlib.sha256(
        "|".join(sorted(document.get("target_languages", []))).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"conversation_rerun_updates_{document['id']}_"
        f"{document.get('bible_revision', 0)}_"
        f"{document.get('glossary_revision', 0)}_{language_signature}"
    )


def request_translation(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    abandon_conversation_rerun(document_id)
    if document.get("conversation_action") == "languages":
        complete_conversation_language_action(document_id)
    if document.get("conversation_action") == "glossary":
        add_message(document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")
    if not document.get("target_languages"):
        document["conversation_action"] = "languages"
        return
    document["conversation_action"] = ""
    queue_message(document_id, False)


def cancel_glossary_before_review(document_id: str) -> None:
    """Treat navigation away from the glossary composer as cancelling its draft."""
    document = st.session_state.documents[document_id]
    abandon_conversation_rerun(document_id)
    if document.get("conversation_action") == "languages":
        complete_conversation_language_action(document_id)
    if document.get("conversation_action") == "glossary":
        document["conversation_action"] = ""
        add_message(document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")


def render_conversation_guided_action(document: dict) -> None:
    """Render the active guided question as the newest chat item."""
    document_id = document["id"]
    action = document.get("conversation_action")
    if action == "languages":
        with st.chat_message("assistant"):
            st.markdown("Which languages should I translate to?")
            st.pills(
                "Target languages",
                TARGET_LANGUAGE_OPTIONS,
                selection_mode="multi",
                default=document.get("target_languages", []),
                key=f"conversation_languages_{document_id}",
                label_visibility="collapsed",
                width="stretch",
                wrap=False,
                on_change=update_conversation_languages,
                args=(document_id,),
            )
    elif action == "rerun":
        update_options = pending_rerun_updates(document)
        with st.chat_message("assistant"):
            st.markdown("Which updated inputs should I apply?")
            selected_updates: list[str] = []
            if update_options:
                st.caption("Updated since the last completed translation · selected automatically")
                selected_updates = st.pills(
                    "Updated inputs",
                    update_options,
                    selection_mode="multi",
                    default=update_options,
                    key=rerun_update_widget_key(document),
                    label_visibility="collapsed",
                    width="stretch",
                )
                st.button(
                    "Start rerun",
                    key=f"apply_conversation_rerun_{document_id}",
                    type="primary",
                    disabled=not bool(selected_updates),
                    on_click=apply_conversation_rerun,
                    args=(document_id,),
                )
            else:
                st.caption(
                    "Nothing has changed since the last completed translation. "
                    "Open localization review for failed values, characters, scenes, context, or QA reruns."
                )
    elif action == "translation_upload":
        upload_key = f"conversation_dialogue_upload_{document_id}_{st.session_state.uploader_version}"
        with st.chat_message("assistant"):
            st.file_uploader(
                "Translation data CSV",
                type=["csv"],
                key=upload_key,
                on_change=handle_dialogue_toolbar_upload,
                args=(upload_key, document_id),
            )
    elif action == "bible_upload":
        upload_key = f"conversation_bible_upload_{document_id}_{document.get('bible_revision', 0)}"
        with st.chat_message("assistant"):
            st.file_uploader(
                "Character Bible CSV",
                type=["csv"],
                key=upload_key,
                on_change=handle_project_bible_toolbar_upload,
                args=(document_id, upload_key),
            )
    elif action == "glossary":
        with st.chat_message("assistant"):
            st.markdown(
                "Upload a glossary CSV, or enter terminology rules in the message field below. "
                "Use `source | language | translation`, one rule per line, then press **Apply glossary**."
            )
            upload_key = f"conversation_glossary_upload_{document_id}_{document.get('glossary_revision', 0)}"
            st.file_uploader(
                "Glossary CSV",
                type=["csv"],
                key=upload_key,
                on_change=handle_project_glossary_upload,
                args=(document_id, upload_key),
            )


def render_onboarding_guided_action() -> None:
    action = st.session_state.get("onboarding_action")
    if action == "languages":
        with st.chat_message("assistant"):
            st.markdown("Which languages should I translate to?")
            st.pills(
                "Target languages",
                TARGET_LANGUAGE_OPTIONS,
                selection_mode="multi",
                default=st.session_state.get("onboarding_target_languages", []),
                key="onboarding_conversation_languages",
                label_visibility="collapsed",
                width="stretch",
                wrap=False,
                on_change=update_onboarding_languages,
            )
    elif action == "translation_upload":
        upload_key = f"onboarding_dialogue_upload_{st.session_state.uploader_version}"
        with st.chat_message("assistant"):
            st.file_uploader(
                "Translation data CSV",
                type=["csv"],
                key=upload_key,
                on_change=handle_dialogue_toolbar_upload,
                args=(upload_key,),
            )
    elif action == "bible_upload":
        upload_key = f"onboarding_conversation_bible_{st.session_state.uploader_version}"
        with st.chat_message("assistant"):
            st.file_uploader(
                "Character Bible CSV",
                type=["csv"],
                key=upload_key,
                on_change=handle_onboarding_bible_toolbar_upload,
                args=(upload_key,),
            )
    elif action == "glossary":
        with st.chat_message("assistant"):
            st.markdown(
                "Upload a glossary CSV, or enter terminology rules in the message field below. "
                "Use `source | language | translation`, one rule per line, then press **Apply glossary**."
            )
            upload_key = f"onboarding_glossary_upload_{st.session_state.uploader_version}"
            st.file_uploader(
                "Glossary CSV",
                type=["csv"],
                key=upload_key,
                on_change=handle_onboarding_glossary_upload,
                args=(upload_key,),
            )


def stage_glossary_draft(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    draft = st.session_state.get(f"glossary_{document_id}", "")
    document["glossary_draft"] = draft
    document["glossary_dirty"] = draft != document.get("glossary_text", "")
    document["glossary_validation_errors"] = []
    document["glossary_confirmation"] = ""


def apply_glossary(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    draft = st.session_state.get(f"glossary_{document_id}", "")
    document["glossary_draft"] = draft
    entries, errors = parse_glossary(draft)
    if errors:
        document["glossary_validation_errors"] = errors
        document["glossary_confirmation"] = ""
        document["glossary_dirty"] = True
        document["glossary_rerun_ready"] = False
        return
    document["glossary_text"] = draft
    document["glossary_dirty"] = False
    document["glossary_validation_errors"] = []
    document["glossary_revision"] = document.get("glossary_revision", 0) + 1
    touch_workspace_item(document, "glossary")
    if document.get("result") is None:
        document["glossary_rerun_ready"] = False
        next_step = " It will be used for the first translation."
    else:
        document["glossary_rerun_ready"] = True
        next_step = " Rerun with current glossary is now available."
    document["glossary_confirmation"] = (
        f"{len(entries)} deterministic terminology rule(s) applied." + next_step
    )


def render_glossary_setup(document: dict, document_id: str) -> None:
    """Keep terminology controls at the top of Translation setup."""
    glossary_key = f"glossary_{document_id}"
    st.markdown("**Glossary & protected terms**")
    st.caption(
        "Use source | language | translation, optionally followed by type | notes. "
        "A single term means do not translate."
    )
    if glossary_key not in st.session_state:
        st.session_state[glossary_key] = document.get(
            "glossary_draft", document.get("glossary_text", "")
        )
    glossary_draft = st.text_area(
        "Terminology rules",
        key=glossary_key,
        height=120,
        placeholder="OpenAI\n會員 | English | member",
        on_change=stage_glossary_draft,
        args=(document_id,),
    )
    document["glossary_draft"] = glossary_draft
    glossary_has_changes = glossary_draft != document.get("glossary_text", "")
    document["glossary_dirty"] = glossary_has_changes
    glossary_activity = bool(
        glossary_draft.strip()
        or str(document.get("glossary_text", "")).strip()
        or document.get("glossary_revision", 0) != document.get("result_glossary_revision", 0)
    )
    glossary_changed = bool(
        document.get("glossary_rerun_ready", False)
        and not document.get("glossary_dirty", False)
        and not document.get("glossary_validation_errors")
    )
    apply_column, rerun_column = st.columns(2) if document.get("result") is not None and glossary_activity else (st, None)
    apply_column.button(
        "Apply glossary",
        key=f"apply_glossary_{document_id}",
        type="primary",
        width="stretch",
        disabled=not glossary_has_changes,
        on_click=apply_glossary,
        args=(document_id,),
        help="Validate and apply the current glossary draft.",
    )
    if rerun_column is not None:
        rerun_column.button(
            "Rerun with current glossary",
            key=f"rerun_translation_{document_id}",
            type="primary" if glossary_changed else "secondary",
            width="stretch",
            disabled=bool(document.get("job_id")) or not glossary_changed,
            on_click=queue_full_rerun,
            args=(document_id,),
            help=(
                "Translate the full dataset again using the newly applied glossary."
                if glossary_changed
                else "Apply a valid glossary change before rerunning the full translation."
            ),
        )
    if document.get("glossary_validation_errors"):
        st.error("\n".join(document["glossary_validation_errors"]))
    elif document.get("glossary_confirmation") and not glossary_has_changes:
        st.success(document["glossary_confirmation"])
    if glossary_has_changes:
        st.caption("Changes are not applied yet. Enter adds a new line; click Apply glossary when ready.")
    elif not document.get("glossary_confirmation"):
        st.caption("Enter adds a new line. Click Apply glossary to validate this glossary.")
    if glossary_changed:
        st.success("Glossary update is ready. Rerun the full translation to apply it to every value.")


def append_run_record(document: dict, status: dict, outcome: str, result=None, error: str = "") -> None:
    metadata = dict(status.get("metadata", {}))
    metrics = getattr(result, "metrics", None)
    document.setdefault("runs", []).append({
        "Started (UTC)": metadata.get("started_at", ""),
        "Finished (UTC)": metadata.get("finished_at", ""),
        "Outcome": outcome,
        "Mode": metadata.get("mode", "translation"),
        "Model": metadata.get("model", ""),
        "Prompt version": metadata.get("prompt_version", ""),
        "Modified by": metadata.get("modified_by", ""),
        "Languages": metadata.get("languages", ""),
        "Columns": metadata.get("columns", ""),
        "Glossary": metadata.get("glossary_entries", 0),
        "Glossary version": metadata.get("glossary_revision", 0),
        "Est. tokens": metadata.get("estimated_input_tokens", 0) + metadata.get("estimated_output_tokens", 0),
        "Est. cost (USD)": metadata.get("estimated_cost_usd", 0.0),
        "API calls": getattr(metrics, "api_calls", 0),
        "Escalated batches": getattr(metrics, "escalated_batches", 0),
        "Coverage": getattr(metrics, "coverage", None),
        "Duration (s)": round(getattr(metrics, "duration_seconds", 0.0), 2),
        "Detail": error,
    })


def initialize_state() -> None:
    defaults = {
        "documents": {},
        "active_document_id": None,
        "uploader_version": 0,
        "onboarding_panel": "",
        "onboarding_messages": [],
        "onboarding_draft": "",
        "onboarding_instructions": "",
        "onboarding_target_languages": [],
        "onboarding_action": "",
        "onboarding_glossary_text": "",
        "pending_character_bible": None,
        "pending_character_bible_name": "",
        "pending_character_bible_hash": "",
        "pending_workspace_item_order": [],
        "pending_bible_editor_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Preserve an active file when this upgrade hot-reloads an older session.
    if not st.session_state.documents and st.session_state.get("source_df") is not None:
        dataframe = st.session_state.source_df
        legacy_id = hashlib.sha256(dataframe.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
        st.session_state.documents[legacy_id] = {
            "id": legacy_id,
            "name": st.session_state.get("active_file_name") or "uploaded.csv",
            "project_name": project_name_from_file(st.session_state.get("active_file_name") or "uploaded.csv"),
            "size": st.session_state.get("active_file_size"),
            "translation_upload_hash": "",
            "dataframe": dataframe,
            "profiles": st.session_state.get("profiles") or profile_columns(dataframe),
            "messages": st.session_state.get("messages") or initial_messages(),
            "result": st.session_state.get("result"),
            "unread": False,
            "job_id": None,
            "last_job_error": "",
            "draft": "",
            "target_languages": [],
            "glossary_text": "",
            "glossary_draft": "",
            "glossary_validation_errors": [],
            "glossary_confirmation": "",
            "runs": [],
            "quality_feedback": {},
            "quality_version": 0,
            "empty_prompt_error": False,
            "applied_glossary_signature": glossary_signature([]),
            "glossary_revision": 0,
            "bible_revision": 0,
            "character_bible_name": "",
            "workspace_item_order": ["translation"],
            "result_glossary_revision": 0,
            "result_bible_revision": 0,
            "result_target_languages": [],
            "glossary_dirty": False,
            "glossary_rerun_ready": False,
            "glossary_ui_version": 2,
            "game_mode": detect_game_schema(dataframe),
            "game_config": asdict(infer_game_config(dataframe)) if detect_game_schema(dataframe) else {},
            "character_bible": (
                default_character_bible(dataframe, infer_game_config(dataframe))
                if detect_game_schema(dataframe) else pd.DataFrame()
            ),
            "review_table": None,
            "review_version": 0,
            "evaluate_style": False,
            "plan_preview": None,
            "plan_preview_error": "",
            "send_bible_warning": "",
            "awaiting_bible_override": False,
            "failure_triage": None,
            "failure_triage_signature": "",
            "failure_triage_version": 0,
            "failure_triage_glossary_revision": 0,
            "failure_triage_bible_revision": 0,
            "failure_triage_error": "",
            "failure_triage_notice": "",
            "failure_resolutions": [],
        }
        st.session_state.active_document_id = legacy_id

    for document in st.session_state.documents.values():
        if document.get("glossary_ui_version") != 2:
            previous_glossary = document.get("glossary_draft", document.get("glossary_text", ""))
            document["glossary_draft"] = previous_glossary
            document["glossary_text"] = ""
            document["glossary_dirty"] = bool(previous_glossary)
            document["glossary_rerun_ready"] = False
            document["glossary_validation_errors"] = []
            document["glossary_confirmation"] = ""
            document["glossary_ui_version"] = 2
        document.setdefault("draft", "")
        document.setdefault("translation_instructions", "")
        document.setdefault("last_job_error", "")
        document.setdefault("target_languages", [])
        document.setdefault("translation_upload_hash", "")
        document.setdefault("project_name", project_name_from_file(document.get("name", "uploaded.csv")))
        document.setdefault("glossary_text", "")
        document.setdefault("runs", [])
        document.setdefault("quality_feedback", {})
        document.setdefault("quality_version", 0)
        document.setdefault("applied_glossary_signature", glossary_signature([]))
        document.setdefault("empty_prompt_error", False)
        document.setdefault("glossary_draft", document.get("glossary_text", ""))
        document.setdefault("glossary_validation_errors", [])
        document.setdefault("glossary_confirmation", "")
        document.setdefault("glossary_revision", 0)
        document.setdefault("bible_revision", 0)
        document.setdefault("character_bible_name", "")
        document.setdefault("workspace_item_order", [])
        document.setdefault("result_glossary_revision", 0)
        if "result_bible_revision" not in document:
            document["result_bible_revision"] = (
                document.get("bible_revision", 0) if document.get("result") is not None else 0
            )
        if "result_target_languages" not in document:
            document["result_target_languages"] = (
                list(document["result"].target_languages) if document.get("result") is not None else []
            )
        document.setdefault("glossary_dirty", False)
        document.setdefault("glossary_rerun_ready", False)
        game_mode = document.setdefault("game_mode", detect_game_schema(document["dataframe"]))
        document.setdefault("game_config", asdict(infer_game_config(document["dataframe"])) if game_mode else {})
        if "character_bible" not in document:
            document["character_bible"] = (
                default_character_bible(document["dataframe"], GameConfig(**document["game_config"]))
                if game_mode else pd.DataFrame()
            )
        document.setdefault("review_table", None)
        document.setdefault("review_version", 0)
        document.setdefault("evaluate_style", False)
        document.setdefault("plan_preview", None)
        document.setdefault("plan_preview_error", "")
        document.setdefault("send_bible_warning", "")
        document.setdefault("awaiting_bible_override", False)
        document.setdefault("game_status_signature", "")
        document.setdefault("failure_triage", None)
        document.setdefault("failure_triage_signature", "")
        document.setdefault("failure_triage_version", 0)
        document.setdefault("failure_triage_glossary_revision", 0)
        document.setdefault("failure_triage_bible_revision", 0)
        document.setdefault("failure_triage_error", "")
        document.setdefault("failure_triage_notice", "")
        document.setdefault("failure_resolutions", [])
        document.setdefault("conversation_action", "")


def collect_completed_jobs() -> bool:
    """Move completed background results into their owning document."""
    changed = False
    manager = background_jobs()
    for document in st.session_state.documents.values():
        job_id = document.get("job_id")
        if not job_id:
            continue
        status = manager.status(job_id)
        if status["state"] in {"running", "cancelling"}:
            continue
        document["job_id"] = None
        changed = True
        if status["state"] == "missing":
            document.pop("pending_resolution_audit", None)
            add_message(
                document,
                "assistant",
                "Translation was interrupted because the app restarted. Please send the request again.",
                notify=True,
            )
            continue
        try:
            result = manager.consume(job_id)
        except (TranslationCancelled, CancelledError):
            document.pop("pending_resolution_audit", None)
            document.pop("running_input_snapshot", None)
            append_run_record(document, status, "Cancelled")
            add_message(document, "assistant", "Translation cancelled safely. No partial output replaced your last completed result.", notify=True)
            continue
        except Exception as error:
            document.pop("pending_resolution_audit", None)
            document.pop("running_input_snapshot", None)
            append_run_record(document, status, "Failed", error=str(error))
            document["last_job_error"] = f"Translation stopped: {error}"
            add_message(
                document,
                "assistant",
                f"Translation stopped safely without changing the source data: {error}",
                notify=True,
            )
            continue
        document["last_job_error"] = ""
        input_snapshot = document.pop("running_input_snapshot", {})
        retry_audit = document.pop("pending_resolution_audit", [])
        if retry_audit:
            remaining_failures = {
                (
                    str(row.get("language", "")),
                    str(row.get("column", "")),
                    str(row.get("source", "")),
                    str(row.get("row_position", "")),
                )
                for _, row in result.failures.iterrows()
            }
            for audit in retry_audit:
                failure = audit["failure"]
                identity = (
                    str(failure.get("language", "")),
                    str(failure.get("column", "")),
                    str(failure.get("source", "")),
                    str(failure.get("row_position", "")),
                )
                if identity not in remaining_failures:
                    document.setdefault("failure_resolutions", []).append({
                        "Resolved (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "Action": audit["action"],
                        "Line": failure.get("line_id", ""),
                        "Language": failure.get("language", ""),
                        "Source": failure.get("source", ""),
                        "Manual translation": "",
                        "Note": audit["note"],
                    })
        document["result"] = result
        touch_workspace_item(document, "result")
        document["failure_triage"] = None
        document["failure_triage_signature"] = ""
        document["failure_triage_error"] = ""
        document["quality_feedback"] = {}
        document["quality_version"] = document.get("quality_version", 0) + 1
        if result.game_config:
            document["review_table"] = build_review_table(
                document["dataframe"], result, document.get("review_table"),
            )
            document["review_version"] = document.get("review_version", 0) + 1
        if status.get("metadata", {}).get("full_glossary_application"):
            document["applied_glossary_signature"] = status["metadata"].get(
                "glossary_signature", glossary_signature([])
            )
            document["result_glossary_revision"] = status["metadata"].get(
                "glossary_revision", document.get("glossary_revision", 0)
            )
            document["glossary_rerun_ready"] = (
                document.get("glossary_revision", 0)
                != document.get("result_glossary_revision", 0)
            )
            document["result_bible_revision"] = input_snapshot.get(
                "bible_revision", document.get("bible_revision", 0)
            )
            document["result_target_languages"] = list(
                input_snapshot.get("target_languages", result.target_languages)
            )
            document.pop("pending_rerun_updates", None)
        append_run_record(document, status, "Completed", result=result)
        add_message(
            document,
            "assistant",
            f"Done. Validation passed and coverage is **{result.metrics.coverage:.1%}**. "
            "Open the localization review workbench to review and download the result.",
            notify=True,
        )
        if result.failures.empty:
            add_message(document, "assistant", "✓ No failed values to retry.", notify=True)
    return changed


@st.fragment(run_every=1)
def watch_background_jobs() -> None:
    """Refresh running jobs without blocking file navigation."""
    if collect_completed_jobs():
        st.rerun()
    manager = background_jobs()
    running = []
    for document in st.session_state.documents.values():
        status = manager.status(document.get("job_id"))
        if status["state"] in {"running", "cancelling"}:
            progress = f"{status['done']}/{status['total']}" if status["total"] else "starting"
            running.append(f"{document['name']}: {progress}")
    if running:
        st.caption("⏳ " + " · ".join(running))


def build_quality_sample(result, sample_size: int = 5) -> list[dict]:
    samples: list[dict] = []
    row_count = len(result.dataframe)
    if not row_count:
        return samples
    positions = list(dict.fromkeys([0, row_count // 2, row_count - 1]))
    for language in result.target_languages:
        suffix = re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_")
        for column in result.source_columns:
            translated_column = f"{column}__{suffix}"
            if translated_column not in result.dataframe:
                continue
            for position in positions:
                source = result.dataframe.iloc[position][column]
                translation = result.dataframe.iloc[position][translated_column]
                if pd.isna(source) or not str(source).strip():
                    continue
                samples.append({
                    "row": position + 1,
                    "column": column,
                    "language": language,
                    "source": str(source),
                    "translation": str(translation),
                })
                if len(samples) >= sample_size:
                    return samples
    return samples


def render_quality_review(document: dict, result) -> None:
    samples = build_quality_sample(result)
    if not samples:
        return
    with st.expander("Quality spot check", expanded=False):
        st.caption("Review a small deterministic sample. This is human feedback, separate from operational coverage.")
        feedback = document.setdefault("quality_feedback", {})
        for index, sample in enumerate(samples):
            with st.container(border=True):
                st.caption(f"Row {sample['row']} · {sample['column']} → {sample['language']}")
                st.markdown(f"**Source:** {sample['source']}  \n**Translation:** {sample['translation']}")
                feedback_key = f"{sample['row']}|{sample['column']}|{sample['language']}"
                choice = st.radio(
                    "Review",
                    ["Not reviewed", "Correct", "Needs revision"],
                    index=["Not reviewed", "Correct", "Needs revision"].index(feedback.get(feedback_key, "Not reviewed")),
                    horizontal=True,
                    key=f"quality_{document['id']}_{document.get('quality_version', 0)}_{index}",
                    label_visibility="collapsed",
                )
                feedback[feedback_key] = choice
        reviewed = [value for value in feedback.values() if value != "Not reviewed"]
        accepted = sum(value == "Correct" for value in reviewed)
        if reviewed:
            st.metric("Human acceptance rate", f"{accepted / len(reviewed):.0%}", f"{accepted}/{len(reviewed)} reviewed")
        else:
            st.caption("No sample has been reviewed yet.")


def render_execution_log(document: dict) -> None:
    runs = document.get("runs", [])
    if not runs:
        return
    with st.expander(f"Execution history ({len(runs)})", expanded=False):
        table = pd.DataFrame(runs).iloc[::-1].reset_index(drop=True)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Est. cost (USD)": st.column_config.NumberColumn(format="$%.5f"),
                "Coverage": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.download_button(
            "Download execution log",
            table.to_csv(index=False).encode("utf-8-sig"),
            file_name="translation_execution_log.csv",
            mime="text/csv",
        )


def failure_table_signature(failures: pd.DataFrame) -> str:
    return hashlib.sha256(failures.to_csv(index=False).encode("utf-8")).hexdigest()


def build_failure_triage(
    failures: pd.DataFrame,
    unified_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an editable failure view using the same visible report columns."""
    report_lookup = {}
    if unified_report is not None and not unified_report.empty:
        report_lookup = {
            (str(row["String ID"]), str(row["Language"])): row
            for _, row in unified_report.iterrows()
        }
    rows: list[dict] = []
    for failure_index, failure in failures.reset_index(drop=True).iterrows():
        line_id = str(failure.get("line_id", ""))
        language = str(failure.get("language", ""))
        matched = report_lookup.get((line_id, language))
        visible = (
            {column: matched.get(column, "") for column in REVIEW_REPORT_COLUMNS}
            if matched is not None
            else {column: "" for column in REVIEW_REPORT_COLUMNS}
        )
        visible.update({
            "String ID": line_id,
            "Language": language,
            "Speaker": str(failure.get("speaker", "")),
            "Original": str(failure.get("source", "")),
            "AI translation": "",
            "Confidence": "Failed",
            "Failure reason": str(failure.get("error", "")),
            "Final translation": "",
            "Resolution note": RESOLUTION_NOTES[0],
            "failure_index": int(failure_index),
            "row_position": failure.get("row_position", None),
            "column": str(failure.get("column", "")),
            "source": str(failure.get("source", "")),
            "error": str(failure.get("error", "")),
            "line_id": line_id,
            "speaker": str(failure.get("speaker", "")),
            "language": language,
        })
        rows.append(visible)
    internal_columns = [
        "failure_index", "row_position", "column", "source", "error",
        "line_id", "speaker", "language",
    ]
    return pd.DataFrame(rows, columns=REVIEW_REPORT_COLUMNS + internal_columns)


def infer_failure_resolution(document: dict, row: pd.Series) -> tuple[str, str]:
    """Infer the action from reviewer changes while preserving their selected audit note."""
    note = str(row.get("Resolution note", RESOLUTION_NOTES[0])).strip()
    if note == "Source text kept intentionally":
        return "Skip", note

    manual = str(row.get("Final translation", "")).strip()
    source = str(row.get("Original", row.get("source", ""))).strip()
    if manual and manual != source:
        return "Manual translation", note

    bible_changed = (
        document.get("bible_revision", 0) > document.get("failure_triage_bible_revision", 0)
    )
    glossary_changed = (
        document.get("glossary_revision", 0) > document.get("failure_triage_glossary_revision", 0)
    )
    if bible_changed and glossary_changed:
        return "Guidance retry", note
    if bible_changed:
        return "Guidance retry", note
    if glossary_changed:
        return "Guidance retry", note
    return "", ""


def failure_triage_readiness(document: dict, table: pd.DataFrame) -> tuple[bool, str, str]:
    """Return whether Apply is allowed plus a user-facing status and severity."""
    manual = table["Final translation"].fillna("").astype(str).str.strip()
    source = table["Original"].fillna("").astype(str).str.strip()
    skip_selected = table["Resolution note"].eq("Source text kept intentionally")
    if ((manual != "") & (manual == source) & ~skip_selected).any():
        return False, "A manual translation must be different from the source text.", "warning"

    inferred = [infer_failure_resolution(document, row) for _, row in table.iterrows()]
    actionable_rows = [(action, note) for action, note in inferred if action]
    actionable = len(actionable_rows)
    if not actionable:
        return (
            False,
            "Edit the Character Bible, apply a glossary change, enter a manual translation, or select Source text kept intentionally to continue.",
            "caption",
        )
    if any(note not in RESOLUTION_NOTES[1:] for _, note in actionable_rows):
        return False, "Choose a Resolution note for every change you want to apply.", "warning"
    return True, f"{actionable} resolution(s) ready to apply.", "success"


def apply_failure_triage(document_id: str) -> None:
    document = st.session_state.documents[document_id]
    result = document.get("result")
    table = document.get("failure_triage")
    document["failure_triage_error"] = ""
    document["failure_triage_notice"] = ""
    if result is None or result.failures.empty or table is None or table.empty:
        document["failure_triage_error"] = "There are no unresolved failures to process."
        return
    ready, readiness_message, _ = failure_triage_readiness(document, table)
    if not ready:
        document["failure_triage_error"] = readiness_message
        return

    resolution_records = document.setdefault("failure_resolutions", [])
    resolved_indices: set[int] = set()
    has_retry = False
    manual_changed = False
    retry_audit = []
    for _, row in table.iterrows():
        action, note = infer_failure_resolution(document, row)
        if not action:
            continue
        failure_index = int(row["failure_index"])
        language = str(row["language"])
        column = str(row["column"])
        source = str(row["Original"])
        if action == "Guidance retry":
            has_retry = True
            retry_audit.append({
                "failure": row.to_dict(),
                "action": "Retry",
                "note": note,
            })
            continue
        if action == "Skip":
            resolved_indices.add(failure_index)
            resolution_records.append({
                "Resolved (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "Action": action,
                "Line": row.get("line_id", ""),
                "Language": language,
                "Source": source,
                "Manual translation": "",
                "Note": note,
            })
            continue

        manual = str(row.get("Final translation", "")).strip()
        translated_column = f"{column}__{safe_column_suffix(language)}"
        if translated_column not in result.dataframe.columns:
            document["failure_triage_error"] = f"Missing output column: {translated_column}."
            return
        row_position = row.get("row_position", None)
        if row_position is not None and not pd.isna(row_position):
            position = int(row_position)
            result.dataframe.at[result.dataframe.index[position], translated_column] = manual
            if document.get("review_table") is not None:
                review_mask = (
                    document["review_table"]["row_position"].eq(position)
                    & document["review_table"]["language"].eq(language)
                )
                document["review_table"].loc[review_mask, "ai_translation"] = manual
                document["review_table"].loc[review_mask, "reviewed_translation"] = manual
                document["review_table"].loc[review_mask, "status"] = "Approved"
                document["review_table"].loc[review_mask, "reviewer_comment"] = note
        else:
            source_mask = document["dataframe"][column].fillna("").astype(str).eq(source)
            result.dataframe.loc[source_mask, translated_column] = manual
        resolved_indices.add(failure_index)
        manual_changed = True
        result.metrics.translated_unique_values += 1
        resolution_records.append({
            "Resolved (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "Action": action,
            "Line": row.get("line_id", ""),
            "Language": language,
            "Source": source,
            "Manual translation": manual,
            "Note": note,
        })

    if resolved_indices:
        result.failures = result.failures.drop(index=list(resolved_indices), errors="ignore").reset_index(drop=True)
        result.metrics.failed_unique_values = len(result.failures)
        document["failure_triage"] = None
        document["failure_triage_signature"] = ""
        document["failure_triage_version"] = document.get("failure_triage_version", 0) + 1
    if manual_changed and result.game_config:
        glossary, _ = parse_glossary(document.get("glossary_text", ""))
        result.qa_issues = build_game_qa(
            document["dataframe"],
            result.dataframe,
            GameConfig(**result.game_config),
            result.target_languages,
            glossary,
        )
        document["review_version"] = document.get("review_version", 0) + 1

    if has_retry and not result.failures.empty:
        document["pending_resolution_audit"] = retry_audit
        document["pending_failed_retry"] = True
        document["failure_triage_notice"] = "Retrying unresolved values with the updated guidance."
    elif resolved_indices:
        document["failure_triage_notice"] = "Resolution applied. No model call was made."


def render_failure_triage(document: dict, result) -> None:
    signature = failure_table_signature(result.failures)
    unified_report = None
    if result.game_config and document.get("review_table") is not None:
        glossary_entries, _ = parse_glossary(document.get("glossary_text", ""))
        cards = build_translation_cards(
            document["dataframe"], result, document["review_table"], glossary_entries
        )
        unified_report = build_unified_review_report(
            document["review_table"], cards, result
        )
    if (
        document.get("failure_triage_signature") != signature
        or document.get("failure_triage") is None
        or document.get("failure_triage_ui_version") != 5
    ):
        document["failure_triage"] = build_failure_triage(
            result.failures, unified_report
        )
        document["failure_triage_signature"] = signature
        document["failure_triage_ui_version"] = 5
        document["failure_triage_version"] = document.get("failure_triage_version", 0) + 1
        document["failure_triage_glossary_revision"] = document.get("glossary_revision", 0)
        document["failure_triage_bible_revision"] = document.get("bible_revision", 0)
    with st.expander(f"Failure triage ({len(result.failures)} unresolved)", expanded=False):
        apply_slot = st.empty()
        status_slot = st.empty()
        st.caption(
            "Edit the Character Bible or glossary, enter a manual translation, or choose Source text kept intentionally. Then select a Resolution note; "
            "the button stays disabled until both the change and its reason are ready."
        )
        disabled_columns = [
            column for column in document["failure_triage"].columns
            if column not in {"Final translation", "Resolution note"}
        ]
        document["failure_triage"] = st.data_editor(
            document["failure_triage"],
            key=f"failure_triage_{document['id']}_{document.get('failure_triage_version', 0)}",
            hide_index=True,
            width="stretch",
            disabled=disabled_columns,
            column_config={
                "failure_index": None,
                "row_position": None,
                "column": None,
                "source": None,
                "error": None,
                "line_id": None,
                "speaker": None,
                "language": None,
                "Final translation": st.column_config.TextColumn("Final translation"),
                "Resolution note": st.column_config.SelectboxColumn(
                    "Resolution note", options=RESOLUTION_NOTES, required=True
                ),
            },
        )
        ready, readiness_message, severity = failure_triage_readiness(
            document, document["failure_triage"]
        )
        with apply_slot:
            st.button(
                "Apply resolution",
                key=f"apply_failure_triage_{document['id']}",
                type="primary" if ready else "secondary",
                width="stretch",
                disabled=bool(document.get("job_id")) or not ready,
                on_click=apply_failure_triage,
                args=(document["id"],),
            )
        with status_slot:
            if document.get("failure_triage_error"):
                st.error(document["failure_triage_error"])
            elif document.get("failure_triage_notice"):
                st.info(document["failure_triage_notice"])
            elif severity == "warning":
                st.warning(readiness_message)
            elif severity == "success":
                st.success(readiness_message)
            else:
                st.caption(readiness_message)
        st.download_button(
            "Download failure report",
            result.failures.to_csv(index=False).encode("utf-8-sig"),
            file_name="translation_failures.csv",
            mime="text/csv",
        )


def render_failure_resolution_history(document: dict) -> None:
    if document.get("failure_resolutions"):
        with st.expander(f"Failure resolution history ({len(document['failure_resolutions'])})"):
            st.dataframe(pd.DataFrame(document["failure_resolutions"]), hide_index=True, width="stretch")


def render_selected_context_inspector(document: dict, review: pd.DataFrame, cards: pd.DataFrame) -> None:
    selected_rows = review[review["selected"].fillna(False)]
    if selected_rows.empty:
        st.caption(
            "Select a translation from the directory to open its Translation Card and line context."
        )
        return

    selected_row = selected_rows.iloc[0]
    selected_position = int(selected_row["row_position"])
    card_language = str(selected_row["language"])

    matching_cards = cards[
        cards["row_position"].eq(selected_position) & cards["language"].eq(card_language)
    ]
    if matching_cards.empty:
        st.warning("The Translation Card for this translation is not available yet.")
        return
    card = matching_cards.iloc[0]
    line_id = str(selected_row.get("line_id") or selected_position + 1)
    st.subheader(f"Context inspector · {line_id} · {card_language}")
    if selected_row.get("context_risk"):
        st.warning(f"Needs context review: {selected_row['context_risk']}")
    with st.container(border=False):
        confidence_column, scene_column, speaker_column, emotion_column, note_column = st.columns(
            [0.8, 0.9, 1.25, 0.9, 1.45]
        )
        confidence_column.caption("Confidence")
        confidence_column.write(f"{card['confidence_level']} · {int(card['confidence'])}%")
        scene_column.caption("Scene")
        scene_column.write(selected_row.get("scene_id") or "Not provided")
        speaker_column.caption("Speaker → Listener")
        speaker_column.write(
            f"{selected_row.get('speaker') or 'Unknown'} → "
            f"{selected_row.get('listener') or 'Unknown'}"
        )
        emotion_column.caption("Emotion")
        emotion_column.write(selected_row.get("emotion") or "Not provided")
        note_column.caption("Developer note")
        note_column.write(selected_row.get("scene_context") or "Not provided")
        if card["ambiguity"]:
            st.error(f"Ambiguity: {card['ambiguity']}")
            st.caption(card["provisional_note"])
        if card["qa_flags"]:
            st.warning(f"QA: {card['qa_flags']}")
        st.caption("Reconstructed context and provenance")
        st.text(card["context_sources"])


def select_game_review_entry(document_id: str, row_index: int) -> None:
    """Keep exactly one translation selected without forcing the dialog to close."""
    document = st.session_state.documents[document_id]
    review = document.get("review_table")
    if review is None or row_index not in review.index:
        return
    review = review.copy()
    review.loc[:, "selected"] = False
    review.at[row_index, "selected"] = True
    document["review_table"] = review


def capture_review_grid_event(document_id: str, component_key: str, event_name: str) -> None:
    """Persist a component trigger before the review fragment renders again."""
    component_state = st.session_state.get(component_key)
    payload = (
        component_state.get(event_name)
        if isinstance(component_state, dict)
        else getattr(component_state, event_name, None)
    )
    if not isinstance(payload, dict):
        return
    document = st.session_state.documents.get(document_id)
    if not document:
        return
    if event_name == "view":
        sort_by = payload.get("sort_by")
        if sort_by in {
            "Story context order", "Needs context first", "QA issues first",
            "Low confidence first",
        }:
            document["review_sort"] = sort_by
    elif event_name == "targeted_rerun":
        document["pending_review_grid_submission"] = payload
        document["pending_review_targeted_rerun"] = True
    else:
        document["pending_review_grid_submission"] = payload


def hide_demo_noise_qa(result) -> None:
    """Hide QA checks that make synthetic Demo Mode output noisy or misleading."""
    if (
        demo_mode_enabled()
        and result is not None
        and not result.qa_issues.empty
        and "type" in result.qa_issues.columns
    ):
        line_ids = result.qa_issues.get("line_id", pd.Series("", index=result.qa_issues.index))
        synthetic_punctuation_test = (
            result.qa_issues["type"].eq("punctuation")
            & line_ids.fillna("").astype(str).str.startswith("qa_")
        )
        demo_noise = (
            result.qa_issues["type"].eq("untranslated_chinese")
            | (result.qa_issues["type"].eq("punctuation") & ~synthetic_punctuation_test)
        )
        result.qa_issues = result.qa_issues[~demo_noise].reset_index(drop=True)


def format_length_field(current_text: str, limit_raw: str) -> str:
    """Render a Length cell as "current / limit", or just the current count with no limit."""
    current = len(current_text or "")
    limit = str(limit_raw or "").strip()
    return f"{current} / {limit}" if limit else str(current)


def build_unified_review_report(
    review: pd.DataFrame,
    cards: pd.DataFrame,
    result,
) -> pd.DataFrame:
    """Return one shared, ordered schema for context, QA, and failure reports."""
    card_lookup = {
        (int(row.row_position), str(row.language)): row
        for row in cards.itertuples()
    } if cards is not None and not cards.empty else {}
    failure_lookup: dict[tuple[int, str], str] = {}
    if result.failures is not None and not result.failures.empty:
        for key, group in result.failures.groupby(["row_position", "language"]):
            failure_lookup[(int(key[0]), str(key[1]))] = "; ".join(
                dict.fromkeys(group["error"].fillna("").astype(str))
            )
    rows: list[dict] = []
    for review_row in review.itertuples():
        key = (int(review_row.row_position), str(review_row.language))
        card = card_lookup.get(key)
        rows.append({
            "String ID": str(review_row.line_id),
            "Language": str(review_row.language),
            "Speaker": str(review_row.speaker),
            "Original": (
                f"Keep source · {review_row.source_text}"
                if str(review_row.status) == "Keep source text"
                else str(review_row.source_text)
            ),
            "Developer note": str(getattr(review_row, "scene_context", "") or ""),
            "AI translation": str(review_row.ai_translation),
            "Final translation": str(review_row.reviewed_translation),
            "Context review": str(review_row.context_risk),
            "Needs context": (
                "Yes" if str(review_row.status) == "Needs context" else ""
            ),
            "QA issue": str(getattr(review_row, "qa_issue", "")),
            "Suggested fix": str(getattr(review_row, "suggested_fix", "")),
            "Length": format_length_field(
                str(review_row.reviewed_translation), getattr(review_row, "character_limit", "")
            ),
            "Confidence": (
                f"{getattr(card, 'confidence_level', '') or 'Unknown'} · {int(getattr(card, 'confidence', 0))}%"
                if card else ""
            ),
            "Failure reason": failure_lookup.get(key, ""),
            "Available context and provenance": (
                str(getattr(card, "context_sources", "")) if card else ""
            ),
            "Resolution note": "",
        })
    return pd.DataFrame(rows, columns=REVIEW_REPORT_COLUMNS)


def render_game_review_actions(document: dict, result, edited: pd.DataFrame, cards: pd.DataFrame) -> None:
    """Render project-wide review tools in the persistent workspace."""
    reviewed_rows = edited[edited["status"].ne("Unreviewed")]
    if not reviewed_rows.empty:
        acceptance = reviewed_rows["status"].isin({"Approved", "Keep source text"}).mean()
        edit_rate = sum(
            1 - SequenceMatcher(None, str(row.ai_translation), str(row.reviewed_translation)).ratio()
            for row in reviewed_rows.itertuples()
        ) / len(reviewed_rows)
        st.caption(
            f"Human acceptance: {acceptance:.0%} · Average reviewer edit distance: {edit_rate:.0%}. "
            "These are workflow signals, not automated truth."
        )
    questions = build_batch_context_questions(cards)
    with st.expander(f"Developer question batch ({len(questions)})", expanded=False):
        if questions.empty:
            st.write("No unresolved context questions were generated.")
        else:
            st.caption(
                "Ambiguities are grouped and deduplicated so the localization manager can ask the developer in one batch."
            )
            st.dataframe(questions, hide_index=True, width="stretch")
            st.download_button(
                "Download developer questions CSV",
                questions.to_csv(index=False).encode("utf-8-sig"),
                file_name="developer_context_questions.csv",
                mime="text/csv",
            )

    unified_report = build_unified_review_report(edited, cards, result)
    context_report = unified_report[
        unified_report["Context review"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if not context_report.empty:
        with st.expander(f"Context review ({len(context_report)})", expanded=False):
            st.caption(
                "Deterministic context risks that may require closer review or a developer question."
            )
            st.dataframe(context_report, hide_index=True, width="stretch")
            st.download_button(
                "Download context review CSV",
                context_report.to_csv(index=False).encode("utf-8-sig"),
                file_name="game_localization_context_review.csv",
                mime="text/csv",
            )

    if not result.qa_issues.empty:
        qa_report = unified_report[
            unified_report["QA issue"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        with st.expander(f"Game QA issues ({len(result.qa_issues)})"):
            st.dataframe(qa_report, hide_index=True, width="stretch")
            st.download_button(
                "Download QA report",
                qa_report.to_csv(index=False).encode("utf-8-sig"),
                file_name="game_localization_qa.csv",
                mime="text/csv",
            )
    if not result.style_evaluations.empty:
        with st.expander(f"AI style evaluation ({len(result.style_evaluations)})"):
            st.caption("Advisory model scores are separate from human approval.")
            st.dataframe(result.style_evaluations, hide_index=True, width="stretch")
@st.fragment
def render_game_review(document: dict, result) -> None:
    """Render review interactions in-place without rebuilding the full dialog."""
    header_placeholder = st.empty()
    hide_demo_noise_qa(result)
    review = build_review_table(
        document["dataframe"],
        result,
        previous=document.get("review_table"),
    )
    document["review_table"] = review
    glossary_entries, _ = parse_glossary(document.get("glossary_text", ""))
    cards = build_translation_cards(
        document["dataframe"], result, review, glossary_entries
    )
    glossary_rules_text = "; ".join(
        f"{entry.source} (preserve)" if entry.preserves_source
        else f"{entry.source} → {entry.translation} ({entry.target_language})"
        for entry in glossary_entries
    ) or "No terminology rules configured."
    review = review.copy()
    card_confidence = {
        (int(row.row_position), str(row.language)): (int(row.confidence), str(row.confidence_level))
        for row in cards.itertuples()
    }
    review["confidence"] = review.apply(
        lambda row: card_confidence.get((int(row["row_position"]), str(row["language"])), (0, ""))[0],
        axis=1,
    )
    review["confidence_level"] = review.apply(
        lambda row: card_confidence.get((int(row["row_position"]), str(row["language"])), (0, ""))[1],
        axis=1,
    )
    edited = review.copy()
    if document.get("review_grid_version") != 3:
        edited.loc[:, "selected"] = False
        document["review_grid_version"] = 3
        document["review_table"] = edited
    cards = build_translation_cards(
        document["dataframe"], result, edited, glossary_entries
    )

    with st.container(border=False, key="review_directory_panel"):
            sort_by = document.get("review_sort", "Story context order")
            all_ordered = edited.sort_values(["row_position", "language"], kind="stable")
            system_context_risk = (
                all_ordered["context_risk"].fillna("").astype(str).str.strip().ne("")
            )
            reviewer_needs_context = all_ordered["status"].eq("Needs context")
            context_review = system_context_risk | reviewer_needs_context
            risk_indexes = set(all_ordered.index[context_review])
            qa_review = all_ordered["qa_flags"].fillna("").astype(str).str.strip().ne("")
            qa_indexes = set(all_ordered.index[qa_review])
            low_confidence_indexes = set(
                all_ordered.index[all_ordered["confidence_level"].eq("Low")]
            )
            failure_lookup: dict[tuple[int, str], str] = {}
            if result.failures is not None and not result.failures.empty:
                for key, group in result.failures.groupby(["row_position", "language"]):
                    failure_lookup[(int(key[0]), str(key[1]))] = "; ".join(
                        dict.fromkeys(group["error"].fillna("").astype(str))
                    )
            editable_indexes = set(all_ordered.index)
            if sort_by == "Needs context first":
                all_ordered = (
                    all_ordered.assign(
                        _needs_context_first=all_ordered.index.isin(risk_indexes)
                    )
                    .sort_values(
                        ["_needs_context_first", "row_position", "language"],
                        ascending=[False, True, True],
                        kind="stable",
                    )
                    .drop(columns="_needs_context_first")
                )
            elif sort_by == "QA issues first":
                all_ordered = (
                    all_ordered.assign(_qa_first=all_ordered.index.isin(qa_indexes))
                    .sort_values(
                        ["_qa_first", "row_position", "language"],
                        ascending=[False, True, True],
                        kind="stable",
                    )
                    .drop(columns="_qa_first")
                )
            elif sort_by == "Low confidence first":
                all_ordered = (
                    all_ordered.assign(
                        _low_confidence_first=all_ordered.index.isin(low_confidence_indexes)
                    )
                    .sort_values(
                        ["_low_confidence_first", "row_position", "language"],
                        ascending=[False, True, True],
                        kind="stable",
                    )
                    .drop(columns="_low_confidence_first")
                )
            if not all_ordered.empty and not all_ordered["selected"].fillna(False).any():
                default_index = all_ordered.index[0]
                edited.loc[:, "selected"] = False
                edited.at[default_index, "selected"] = True
                all_ordered.at[default_index, "selected"] = True
                document["review_table"] = edited
            ordered = all_ordered.copy()
            navigation = pd.DataFrame({
                "Rerun": ordered["rerun_selected"].fillna(False).astype(bool),
                "Details": ordered["selected"].fillna(False).astype(bool),
                "String ID": ordered.apply(
                    lambda row: str(
                        row.get("line_id") or int(row["row_position"]) + 1
                    ),
                    axis=1,
                ),
                "Language": ordered["language"].fillna(""),
                "Speaker": ordered["speaker"].fillna(""),
                "Original": ordered["source_text"].fillna(""),
                "AI translation": ordered.apply(
                    lambda row: (
                        "" if (int(row["row_position"]), str(row["language"])) in failure_lookup
                        else str(row.get("ai_translation") or "")
                    ),
                    axis=1,
                ),
                "Final translation": ordered["reviewed_translation"].fillna(""),
                "Keep source text": ordered["status"].eq("Keep source text"),
                "Context review": ordered["context_risk"].fillna(""),
                "Needs context": ordered["status"].eq("Needs context"),
                "QA issue": ordered["qa_issue"].fillna(""),
                "Suggested fix": ordered["suggested_fix"].fillna(""),
                "Confidence score": ordered.apply(
                    lambda row: f"{int(row.get('confidence') or 0)}%",
                    axis=1,
                ),
                "Confidence level": ordered["confidence_level"].fillna(""),
                "Failure reason": ordered.apply(
                    lambda row: failure_lookup.get(
                        (int(row["row_position"]), str(row["language"])), ""
                    ),
                    axis=1,
                ),
                "Notes": ordered["reviewer_comment"].fillna(""),
            })

            submitted = None
            if navigation.empty:
                st.info("No translations are available for review.")
                updated_navigation = navigation
            else:
                index_by_key = {str(index): index for index in navigation.index}
                cards_by_key = {
                    (int(card.row_position), str(card.language)): card
                    for card in cards.itertuples()
                }
                component_rows = []
                for index, row in navigation.iterrows():
                    review_row = ordered.loc[index]
                    card = cards_by_key.get(
                        (int(review_row["row_position"]), str(review_row["language"]))
                    )
                    component_rows.append({
                        "row_key": str(index),
                        "rerun": bool(row["Rerun"]),
                        "details": bool(row["Details"]),
                        "label": f"{row['String ID']} · {row['Language']}",
                        "string_id": str(row["String ID"] or ""),
                        "language": str(row["Language"] or ""),
                        "speaker": str(row["Speaker"] or ""),
                        "original": str(row["Original"] or ""),
                        "ai_translation": str(row["AI translation"] or ""),
                        "confidence_score": str(row["Confidence score"] or ""),
                        "confidence_level": str(row["Confidence level"] or ""),
                        "confidence": (
                            f"{row['Confidence level'] or 'Unknown'} · "
                            f"{row['Confidence score'] or '0%'}"
                        ),
                        "context_review": str(row["Context review"] or ""),
                        "qa_issue": str(row["QA issue"] or ""),
                        "suggested_fix": str(row["Suggested fix"] or ""),
                        "failure_reason": str(row["Failure reason"] or ""),
                        "translation": str(row["Final translation"] or ""),
                        "notes": str(row["Notes"] or ""),
                        "needs_context": bool(row["Needs context"]),
                        "keep_source": bool(row["Keep source text"]),
                        "context_risk": index in risk_indexes,
                        "low_confidence": index in low_confidence_indexes,
                        "editable": index in editable_indexes,
                        "scene": str(review_row.get("scene_id") or "Not provided"),
                        "speaker_listener": (
                            f"{review_row.get('speaker') or 'Unknown'} → "
                            f"{review_row.get('listener') or 'Unknown'}"
                        ),
                        "emotion": str(review_row.get("emotion") or "Not provided"),
                        "developer_note": str(review_row.get("scene_context") or ""),
                        "character_limit": str(review_row.get("character_limit") or ""),
                        "screenshot": str(review_row.get("screenshot") or ""),
                        "placeholder_details": str(review_row.get("placeholder_details") or ""),
                        "glossary_hits": str(getattr(card, "glossary_hits", "") if card else ""),
                        "tm_match": str(getattr(card, "tm_match", "") if card else ""),
                        "glossary_rules": glossary_rules_text,
                        "previous_lines": str(review_row.get("previous_lines") or ""),
                        "next_lines": str(review_row.get("next_lines") or ""),
                        "context_sources": str(
                            getattr(card, "context_sources", "") if card else ""
                        ),
                    })
                component_key = (
                    f"review_grid_component_{document['id']}_"
                    f"{document.get('review_version', 0)}"
                )
                component_result = REVIEW_GRID_COMPONENT(
                    key=component_key,
                    data={
                        "rows": component_rows,
                        "storage_key": (
                            f"{document['id']}:{document.get('review_version', 0)}:"
                            f"grid-{document.get('review_grid_version', 3)}"
                        ),
                        "sort_by": sort_by,
                        "characters": sorted(
                            value for value in review["speaker"].dropna().astype(str).unique()
                            if value
                        ),
                        "scenes": sorted(
                            value for value in review["scene_id"].dropna().astype(str).unique()
                            if value
                        ),
                        "has_failures": not result.failures.empty,
                        "failed_count": len(result.failures),
                        "has_needs_context": bool(risk_indexes),
                        "needs_context_count": len(risk_indexes),
                        "has_qa_flagged": bool(review["qa_flags"].fillna("").ne("").any()),
                        "qa_flagged_count": int(review["qa_flags"].fillna("").ne("").sum()),
                    },
                    width="stretch",
                    height="content",
                    on_submitted_change=lambda: capture_review_grid_event(
                        document["id"], component_key, "submitted"
                    ),
                    on_targeted_rerun_change=lambda: capture_review_grid_event(
                        document["id"], component_key, "targeted_rerun"
                    ),
                    on_view_change=lambda: capture_review_grid_event(
                        document["id"], component_key, "view"
                    ),
                )
                updated_navigation = navigation.copy()
                submitted = document.pop("pending_review_grid_submission", None)
                if not isinstance(submitted, dict):
                    submitted = getattr(component_result, "submitted", None)
                if isinstance(submitted, dict):
                    for submitted_row in submitted.get("rows", []):
                        index = index_by_key.get(str(submitted_row.get("row_key", "")))
                        if index is None:
                            continue
                        updated_navigation.at[index, "Rerun"] = bool(
                            submitted_row.get("rerun", False)
                        )
                        updated_navigation.at[index, "Details"] = bool(
                            submitted_row.get("details", False)
                        )
                        if index not in editable_indexes:
                            continue
                        updated_navigation.at[index, "Final translation"] = str(
                            submitted_row.get("translation", "")
                        )
                        updated_navigation.at[index, "Notes"] = str(
                            submitted_row.get("notes", "")
                        )
                        updated_navigation.at[index, "Needs context"] = bool(
                            submitted_row.get("needs_context", False)
                        )
                        updated_navigation.at[index, "Keep source text"] = bool(
                            submitted_row.get("keep_source", False)
                        )

            previous_selected = set(edited.index[edited["selected"].fillna(False)])
            selected_candidates = [
                index for index in updated_navigation.index
                if bool(updated_navigation.at[index, "Details"])
            ]
            newly_selected = [index for index in selected_candidates if index not in previous_selected]
            selected_index = (
                newly_selected[-1] if newly_selected
                else selected_candidates[0] if selected_candidates
                else None
            )
            edited.loc[:, "selected"] = False
            if selected_index is not None:
                edited.at[selected_index, "selected"] = True
            for index in updated_navigation.index:
                edited.at[index, "rerun_selected"] = bool(
                    updated_navigation.at[index, "Rerun"]
                )

            blocked_change = False
            for index in updated_navigation.index:
                old_status = str(edited.at[index, "status"] or "Unreviewed")
                translation = str(updated_navigation.at[index, "Final translation"] or "")
                reviewer_comment = str(updated_navigation.at[index, "Notes"] or "")
                translation_changed = translation != str(
                    edited.at[index, "reviewed_translation"] or ""
                )
                note_changed = reviewer_comment != str(
                    edited.at[index, "reviewer_comment"] or ""
                )
                needs_context = bool(updated_navigation.at[index, "Needs context"])
                keep_source = bool(updated_navigation.at[index, "Keep source text"])
                if index not in editable_indexes:
                    blocked_change = blocked_change or any([
                        translation_changed,
                        note_changed,
                        needs_context != (old_status == "Needs context"),
                        keep_source != (old_status == "Keep source text"),
                    ])
                    continue
                if translation_changed:
                    edited.at[index, "reviewed_translation"] = translation
                if note_changed:
                    edited.at[index, "reviewer_comment"] = reviewer_comment
                if needs_context and keep_source:
                    old_needs_context = old_status == "Needs context"
                    old_keep_source = old_status == "Keep source text"
                    if keep_source and not old_keep_source:
                        needs_context = False
                    elif needs_context and not old_needs_context:
                        keep_source = False
                    else:
                        needs_context = False
                if keep_source:
                    edited.at[index, "status"] = "Keep source text"
                    edited.at[index, "reviewed_translation"] = edited.at[index, "source_text"]
                elif needs_context:
                    edited.at[index, "status"] = "Needs context"
                elif old_status in {"Needs context", "Keep source text"}:
                    edited.at[index, "status"] = (
                        "Approved" if translation_changed else "Unreviewed"
                    )
                elif translation_changed:
                    edited.at[index, "status"] = "Approved"

            document["review_table"] = edited
            if blocked_change:
                st.warning("That review change could not be applied. Reload the workbench and try again.")
            if document.pop("pending_review_targeted_rerun", False):
                rerun_scope = (
                    str(submitted.get("scope", "Selected lines"))
                    if isinstance(submitted, dict) else "Selected lines"
                )
                rerun_value = (
                    str(submitted.get("value", ""))
                    if isinstance(submitted, dict) else ""
                )
                if rerun_scope == "Selected lines":
                    rerun_positions = sorted(set(
                        edited.loc[
                            edited["rerun_selected"].fillna(False), "row_position"
                        ].astype(int)
                    ))
                elif rerun_scope == "Failed values":
                    failure_positions = (
                        pd.to_numeric(result.failures["row_position"], errors="coerce").dropna()
                        if "row_position" in result.failures.columns
                        else pd.Series(dtype="float64")
                    )
                    rerun_positions = sorted(set(failure_positions.astype(int)))
                elif rerun_scope == "Needs context":
                    needs_context_indexes = risk_indexes | set(
                        edited.index[edited["status"].eq("Needs context")]
                    )
                    rerun_positions = sorted(set(
                        edited.loc[
                            edited.index.isin(needs_context_indexes), "row_position"
                        ].astype(int)
                    ))
                else:
                    rerun_positions = select_review_rows(
                        edited, rerun_scope, rerun_value
                    )
                if rerun_positions:
                    edited.loc[:, "rerun_selected"] = False
                    document["review_table"] = edited
                    document["pending_game_rerun"] = {
                        "scope": rerun_scope,
                        "value": rerun_value,
                        "row_positions": rerun_positions,
                    }
                    add_message(
                        document,
                        "user",
                        f"Targeted rerun · {rerun_scope}"
                        + (f" · {rerun_value}" if rerun_value else "")
                        + f": {len(rerun_positions)} line(s)",
                    )
                    st.rerun()

    with header_placeholder.container():
        title_column, summary_column, download_column = st.columns(
            [2.1, 5.4, 1], vertical_alignment="center"
        )
        with title_column:
            st.markdown("### Localization review workbench")
        with summary_column:
            st.caption(
                f"{len(navigation)} row(s) · "
                f"{int(system_context_risk.sum())} context review suggested · "
                f"{int(reviewer_needs_context.sum())} marked Needs context · "
                "all rows are editable · "
                + (
                    "Story context order keeps neighboring dialogue interleaved"
                    if sort_by == "Story context order"
                    else f"Sorted by {sort_by.lower()}"
                )
            )
        with download_column:
            with st.container(key="review_header_download"):
                st.download_button(
                    "Download CSV",
                    reviewed_export(
                        document["dataframe"],
                        result,
                        document["review_table"],
                        glossary_entries,
                    )
                    .to_csv(index=False)
                    .encode("utf-8-sig"),
                    file_name="reviewed_game_localization.csv",
                    mime="text/csv",
                    width="content",
                )

@st.dialog(" ", width="large", on_dismiss="rerun")
def localization_review_drawer(document_id: str) -> None:
    """Open the translation-card browser in a spacious drawer."""
    document = st.session_state.documents[document_id]
    result = document.get("result")
    if result is None or not result.game_config:
        st.info("Run a game localization translation before opening the review workbench.")
        return
    review = document.get("review_table")
    if review is None:
        review = build_review_table(document["dataframe"], result)
        document["review_table"] = review
    render_game_review(document, result)


def build_generic_review_table(result) -> pd.DataFrame:
    """Normalize a general CSV result into the same review-oriented reading model."""
    failure_keys = set()
    if result.failures is not None and not result.failures.empty:
        failure_keys = {
            (str(row.get("column", "")), str(row.get("language", "")), str(row.get("source", "")))
            for _, row in result.failures.iterrows()
        }
    rows: list[dict] = []
    for position, (_, source_row) in enumerate(result.dataframe.iterrows(), start=1):
        for source_column in result.source_columns:
            original = "" if pd.isna(source_row[source_column]) else str(source_row[source_column])
            if not original.strip():
                continue
            for language in result.target_languages:
                translated_column = f"{source_column}__{safe_column_suffix(language)}"
                translation = (
                    "" if translated_column not in result.dataframe.columns or pd.isna(source_row[translated_column])
                    else str(source_row[translated_column])
                )
                failed = (source_column, language, original) in failure_keys
                rows.append({
                    "Row": position,
                    "Source column": source_column,
                    "Language": language,
                    "Original": original,
                    "AI translation": translation,
                    "Status": "Failed" if failed else "Translated",
                })
    return pd.DataFrame(rows)


@st.dialog(" ", width="large", on_dismiss="rerun")
def generic_translation_review_drawer(document_id: str) -> None:
    """Review and download a non-game translation without returning to the legacy preview."""
    document = st.session_state.documents[document_id]
    result = document.get("result")
    if result is None:
        st.info("Run a translation before opening the review workbench.")
        return
    title_column, summary_column, download_column = st.columns(
        [2.2, 4.8, 1], vertical_alignment="center"
    )
    title_column.markdown("### Localization review workbench")
    summary_column.caption(
        f"{len(result.dataframe):,} source row(s) · "
        f"{len(result.source_columns)} translated column(s) · "
        f"{len(result.target_languages)} target language(s)"
    )
    with download_column:
        st.download_button(
            "Download CSV",
            result.dataframe.to_csv(index=False).encode("utf-8-sig"),
            file_name="translated.csv",
            mime="text/csv",
            width="content",
        )
    review_table = build_generic_review_table(result)
    st.dataframe(
        review_table,
        hide_index=True,
        width="stretch",
        height=min(680, 42 + max(len(review_table), 1) * 35),
        column_config={
            "Row": st.column_config.NumberColumn("Row", format="%d"),
            "Source column": st.column_config.TextColumn("Source column"),
            "Language": st.column_config.TextColumn("Language"),
            "Original": st.column_config.TextColumn("Original", width="large"),
            "AI translation": st.column_config.TextColumn("AI translation", width="large"),
            "Status": st.column_config.TextColumn("Status"),
        },
    )
    render_quality_review(document, result)


def render_game_review_launcher(document: dict, result) -> None:
    hide_demo_noise_qa(result)
    review = build_review_table(
        document["dataframe"],
        result,
        previous=document.get("review_table"),
    )
    document["review_table"] = review
    glossary_entries, _ = parse_glossary(document.get("glossary_text", ""))
    cards = build_translation_cards(
        document["dataframe"], result, review, glossary_entries
    )
    st.markdown("**Localization review workbench**")
    context_review = int(
        review["context_risk"].fillna("").astype(str).str.strip().ne("").sum()
    )
    confidence_counts = cards["confidence_level"].value_counts() if not cards.empty else {}
    low_confidence = int(confidence_counts.get("Low", 0))
    failed_values = len(result.failures)
    first, second, third, fourth = st.columns(4)
    first.metric("Low confidence", low_confidence)
    second.metric("Context review", context_review)
    third.metric("QA issues", len(result.qa_issues))
    fourth.metric("Failed value", failed_values)
    st.caption(
        f"Translation cards: {confidence_counts.get('High', 0)} high · "
        f"{confidence_counts.get('Medium', 0)} medium · "
        f"{confidence_counts.get('Low', 0)} low confidence · "
        f"{len(approved_translation_memory(review))} project TM entrie(s)."
    )
    render_game_review_actions(document, result, review, cards)


def render_generic_review_launcher(result) -> None:
    """Keep the non-game result in the same persistent workspace hierarchy."""
    st.markdown("**Localization review workbench**")
    first, second, third, fourth = st.columns(4)
    first.metric("Source rows", len(result.dataframe))
    second.metric("Source columns", len(result.source_columns))
    third.metric("Languages", len(result.target_languages))
    fourth.metric("Failed value", len(result.failures))
    st.caption(
        "Open the review workbench from Conversation to compare original and translated values, "
        "run a quality spot check, and download the CSV."
    )


def render_result(result, document: dict) -> None:
    if result is None:
        return
    st.subheader(
        "Translation result",
        help="The latest completed translation will stay here, independent of the conversation.",
    )
    metrics = result.metrics
    first, second, third, fourth = st.columns(4)
    first.metric("Rows", f"{metrics.rows:,}")
    second.metric("Coverage", f"{metrics.coverage:.1%}")
    third.metric("API calls", metrics.api_calls)
    fourth.metric("Retries / splits", f"{metrics.retries} / {metrics.fallback_splits}")
    st.caption(
        "Validation passed: row count and all original columns/values are unchanged. "
        f"Translated {metrics.translated_unique_values:,} of {metrics.requested_unique_values:,} unique values."
    )
    if metrics.escalated_batches:
        st.caption(f"Model routing: {metrics.escalated_batches} batch(es) escalated to the stronger model.")
    if metrics.state_events:
        with st.expander(f"Orchestration state trace ({len(metrics.state_events)} events)", expanded=False):
            st.caption(
                "Each entry is a transition through translate → validate → (success | retry | split) → "
                "retain_source, in the order this run actually took. See BATCH_STATE_GRAPH in translation_agent.py."
            )
            st.code(" → ".join(metrics.state_events), language=None)

def render_conversation_result_actions(
    document: dict, result, primary_action=None, retry_column=None
) -> None:
    """Keep completed-result actions beside the conversation that initiated the job."""
    if primary_action is None or retry_column is None:
        primary_action, retry_column = st.columns(2)
    with primary_action:
        if result.game_config:
            if st.button(
                "Open localization review",
                key=f"open_game_review_{document['id']}",
                icon=":material/table_view:",
                type="primary",
                width="stretch",
                on_click=cancel_glossary_before_review,
                args=(document["id"],),
            ):
                localization_review_drawer(document["id"])
        else:
            if st.button(
                "Open localization review",
                key=f"open_generic_review_{document['id']}",
                icon=":material/table_view:",
                type="primary",
                width="stretch",
                on_click=cancel_glossary_before_review,
                args=(document["id"],),
            ):
                generic_translation_review_drawer(document["id"])
    with retry_column:
        st.button(
            "Targeted rerun",
            key=f"open_targeted_rerun_{document['id']}",
            icon=":material/replay:",
            type="primary" if pending_rerun_updates(document) else "secondary",
            width="stretch",
            disabled=bool(document.get("job_id")),
            on_click=open_conversation_action,
            args=(document["id"], "rerun"),
            help="Apply Character Bible, target-language, or glossary updates made after the last translation.",
        )


def render_generic_conversation_action_tray(document: dict, active_job: dict) -> None:
    """Use the current Conversation controls for a non-game CSV without game-only actions."""
    document_id = document["id"]
    has_result = document.get("result") is not None
    has_languages = bool(document.get("target_languages"))
    first_row = st.columns(3)

    with first_row[0], st.container(key="translation_upload_action"):
        st.button(
            "Update translation data",
            key=f"open_translation_upload_{document_id}",
            type="secondary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=active_job["state"] in {"running", "cancelling"},
            help="Replace the source CSV while keeping its glossary, target languages, and conversation.",
            on_click=open_conversation_action,
            args=(document_id, "translation_upload"),
        )
    with first_row[1]:
        st.button(
            "Add language" if has_result else "Target languages ✓" if has_languages else "Target languages",
            key=f"open_conversation_languages_{document_id}",
            type="secondary" if has_languages else "primary",
            icon=":material/language:",
            width="stretch",
            on_click=open_conversation_action,
            args=(document_id, "languages"),
        )
    with first_row[2]:
        st.button(
            "Update glossary"
            if str(document.get("glossary_text", "")).strip()
            else "Add glossary" if has_result else "Glossary",
            key=f"open_conversation_glossary_{document_id}",
            icon=":material/book_2:",
            width="stretch",
            on_click=open_conversation_action,
            args=(document_id, "glossary"),
        )

    if has_result:
        second_row = st.columns(2)
        render_conversation_result_actions(
            document, document["result"],
            primary_action=second_row[0], retry_column=second_row[1],
        )
    elif has_languages:
        second_row = st.columns(2)
        with second_row[0]:
            st.button(
                "Preview cost",
                key=f"preview_translation_plan_{document_id}",
                width="stretch",
                disabled=active_job["state"] in {"running", "cancelling"},
                on_click=preview_translation_plan,
                args=(document_id,),
            )
        with second_row[1]:
            st.button(
                "Translate",
                key=f"translate_{document_id}",
                type="primary",
                icon=":material/translate:",
                width="stretch",
                disabled=active_job["state"] in {"running", "cancelling"},
                on_click=request_translation,
                args=(document_id,),
            )


def render_conversation_action_tray(document: dict, active_job: dict) -> None:
    """Render stable project actions above the pinned native chat input."""
    if not document.get("game_mode"):
        render_generic_conversation_action_tray(document, active_job)
        return
    document_id = document["id"]
    has_translation_result = document.get("result") is not None
    current_config = (
        GameConfig(**document["game_config"])
        if document.get("game_mode") and document.get("game_config") else GameConfig()
    )
    target_languages_ready = bool(document.get("target_languages"))
    character_bible_ready = (
        not document.get("game_mode") or not missing_bible_speakers(document)
    )
    setup_ready = target_languages_ready and character_bible_ready
    use_six_button_layout = has_translation_result or setup_ready
    first_row = st.columns(3 if use_six_button_layout else 2)
    second_row = st.columns(3 if use_six_button_layout else 2)
    if has_translation_result:
        bible_action, language_action, glossary_action = first_row
        upload_action = second_row[0]
    elif setup_ready:
        upload_action, bible_action, language_action = first_row
        glossary_action = second_row[0]
    else:
        upload_action, bible_action = first_row
        language_action, glossary_action = second_row
    with upload_action, st.container(key="translation_upload_action"):
        st.button(
            "Update translation data",
            key=f"open_translation_upload_{document_id}",
            type="secondary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=active_job["state"] in {"running", "cancelling"},
            help="Replace the source CSV in this project while keeping its Bible, glossary, and target languages.",
            on_click=open_conversation_action,
            args=(document_id, "translation_upload"),
        )
    with bible_action, st.container(key="bible_upload_action"):
        st.button(
            "Update Bible" if document.get("bible_revision", 0) > 0 else "Upload Character Bible",
            key=f"open_bible_upload_{document_id}",
            type="secondary" if document.get("bible_revision", 0) > 0 else "primary",
            icon=":material/menu_book:",
            width="stretch",
            disabled=not document.get("game_mode") or not bool(current_config.speaker),
            help=(
                "This file has no speaker column, so a Character Bible is not required."
                if not document.get("game_mode") or not current_config.speaker else None
            ),
            on_click=open_conversation_action,
            args=(document_id, "bible_upload"),
        )
    with language_action:
        st.button(
            (
                "Add language"
                if has_translation_result
                else "Target languages ✓" if document.get("target_languages") else "Target languages"
            ),
            key=f"open_conversation_languages_{document_id}",
            type="secondary" if document.get("target_languages") else "primary",
            icon=":material/language:",
            width="stretch",
            on_click=open_conversation_action,
            args=(document_id, "languages"),
        )
    with glossary_action:
        st.button(
            (
                "Update glossary"
                if str(document.get("glossary_text", "")).strip()
                else "Add glossary" if has_translation_result else "Glossary"
            ),
            key=f"open_conversation_glossary_{document_id}",
            icon=":material/book_2:",
            width="stretch",
            on_click=open_conversation_action,
            args=(document_id, "glossary"),
        )

    if setup_ready:
        if document.get("game_mode"):
            if not has_translation_result:
                with second_row[1]:
                    st.button(
                        "Preview cost",
                        key=f"preview_game_plan_{document_id}",
                        width="stretch",
                        disabled=active_job["state"] in {"running", "cancelling"},
                        on_click=preview_translation_plan,
                        args=(document_id,),
                    )
            if not has_translation_result:
                with second_row[2]:
                    st.button(
                        "Translate",
                        key=f"translate_{document_id}",
                        type="primary",
                        icon=":material/translate:",
                        width="stretch",
                        disabled=active_job["state"] in {"running", "cancelling"},
                        on_click=request_translation,
                        args=(document_id,),
                    )
        elif not has_translation_result:
            with second_row[2]:
                st.button(
                    "Translate",
                    key=f"translate_{document_id}",
                    type="primary",
                    icon=":material/translate:",
                    width="stretch",
                    disabled=active_job["state"] in {"running", "cancelling"},
                    on_click=request_translation,
                    args=(document_id,),
                )
    if has_translation_result:
        render_conversation_result_actions(
            document,
            document["result"],
            primary_action=second_row[1],
            retry_column=second_row[2],
        )


def render_post_setup_result(result, document: dict) -> None:
    """Render review content after Translation setup without moving the setup itself."""
    if result is None:
        return
    render_failure_resolution_history(document)
    if not result.failures.empty:
        render_failure_triage(document, result)
    render_execution_log(document)
    if result.game_config:
        render_game_review_launcher(document, result)
    else:
        render_generic_review_launcher(result)


def open_new_project_upload() -> None:
    """Return to the Conversation upload step without discarding open projects."""
    save_active_draft()
    active_id = st.session_state.get("active_document_id")
    document = st.session_state.get("documents", {}).get(active_id)
    if document and document.get("conversation_action") == "rerun":
        abandon_conversation_rerun(active_id)
    if document and document.get("conversation_action") == "languages":
        complete_conversation_language_action(active_id)
    if document and document.get("conversation_action") == "glossary":
        document["conversation_action"] = ""
        add_message(document, "assistant", "Glossary editing cancelled. No terminology rules were changed.")
    st.session_state.active_document_id = None


def show_onboarding_panel(panel: str) -> None:
    st.session_state.onboarding_panel = panel


def read_uploaded_character_bible(uploaded) -> tuple[pd.DataFrame | None, str]:
    try:
        imported = read_csv(uploaded)
    except Exception as error:
        return None, f"I couldn’t read this Character Bible CSV. Details: {error}"
    errors = validate_character_bible(imported)
    if errors:
        return None, "This file does not look like a Character Bible: " + "; ".join(errors)
    return imported.fillna(""), ""


def translation_data_type_error(dataframe: pd.DataFrame) -> str:
    """Reject a Character Bible accidentally submitted as translation data."""
    normalized = {str(column).strip().lower() for column in dataframe.columns}
    bible_guidance = set(CHARACTER_BIBLE_COLUMNS[1:])
    source_aliases = {
        "source_text", "source", "dialogue", "text", "en", "english", "ja", "japanese"
    }
    looks_like_bible = (
        "speaker" in normalized
        and bool(normalized & bible_guidance)
        and not bool(normalized & source_aliases)
    )
    if looks_like_bible:
        return (
            "This file looks like a Character Bible, not translation data. "
            "Upload it with **Upload Character Bible**. The current translation project was not changed."
        )
    return ""


def read_uploaded_glossary(uploaded) -> tuple[str, str]:
    """Convert a glossary CSV into the same deterministic text format as manual input."""
    try:
        imported = read_csv(uploaded).fillna("")
    except Exception as error:
        return "", f"I couldn’t read this glossary CSV. Details: {error}"
    normalized = {str(column).strip().lower(): column for column in imported.columns}
    source_column = normalized.get("source") or normalized.get("term")
    if source_column is None:
        return "", "This glossary CSV needs a `source` (or `term`) column."
    lines: list[str] = []
    for row_number, row in imported.iterrows():
        source = str(row.get(source_column, "")).strip()
        if not source:
            continue
        language = str(row.get(normalized.get("language", ""), "")).strip()
        translation = str(row.get(normalized.get("translation", ""), "")).strip()
        if language or translation:
            if not language or not translation:
                return "", f"Row {row_number + 2}: language and translation must both be provided."
            term_type = str(row.get(normalized.get("type", ""), "")).strip()
            notes = str(row.get(normalized.get("notes", ""), "")).strip()
            parts = [source, language, translation]
            if term_type or notes:
                parts.extend([term_type, notes])
            lines.append(" | ".join(parts))
        else:
            lines.append(source)
    text = "\n".join(lines)
    _, errors = parse_glossary(text)
    return ("", "; ".join(errors)) if errors else (text, "")


def handle_project_glossary_upload(document_id: str, upload_key: str) -> None:
    uploaded = st.session_state.get(upload_key)
    if uploaded is None:
        return
    text, error = read_uploaded_glossary(uploaded)
    document = st.session_state.documents[document_id]
    if error:
        add_message(document, "assistant", error)
        document["glossary_validation_errors"] = [error]
        return
    st.session_state[f"glossary_{document_id}"] = text
    document["glossary_draft"] = text
    apply_glossary(document_id)
    add_message(document, "user", f"Uploaded glossary: **{uploaded.name}**")
    add_message(document, "assistant", "Glossary CSV validated and applied.")
    document["conversation_action"] = ""


def handle_onboarding_glossary_upload(upload_key: str) -> None:
    uploaded = st.session_state.get(upload_key)
    if uploaded is None:
        return
    text, error = read_uploaded_glossary(uploaded)
    if error:
        st.session_state.onboarding_messages = [
            *st.session_state.get("onboarding_messages", []),
            {"role": "assistant", "content": error},
        ]
        return
    st.session_state.onboarding_glossary_text = text
    st.session_state["pending_glossary_editor"] = text
    touch_pending_workspace_item("glossary")
    st.session_state.onboarding_action = ""
    st.session_state.onboarding_messages = [
        *st.session_state.get("onboarding_messages", []),
        {"role": "user", "content": f"Uploaded glossary: **{uploaded.name}**"},
        {"role": "assistant", "content": "Glossary CSV validated and applied."},
    ]


def handle_dialogue_toolbar_upload(
    upload_key: str, source_document_id: str | None = None
) -> None:
    """Create an onboarding project or replace data inside the active project."""
    uploaded = st.session_state.get(upload_key)
    if uploaded is not None:
        if source_document_id:
            source_document = st.session_state.get("documents", {}).get(source_document_id)
            if source_document:
                source_document["conversation_action"] = ""
                update_project_translation_data(source_document_id, uploaded)
        else:
            st.session_state.onboarding_action = ""
            create_uploaded_project(
                uploaded,
                conversation_lead="Please upload the translation-data CSV you want to open.",
            )


def update_project_translation_data(document_id: str, uploaded) -> bool:
    """Replace one project's source package without creating a sidebar project."""
    document = st.session_state.get("documents", {}).get(document_id)
    if document is None:
        return False
    try:
        dataframe = read_csv(uploaded)
    except Exception as error:
        add_message(document, "user", f"Uploaded translation data: **{uploaded.name}**")
        add_message(document, "assistant", f"I couldn’t read that CSV: {error}")
        return False
    if dataframe.empty:
        add_message(document, "user", f"Uploaded translation data: **{uploaded.name}**")
        add_message(document, "assistant", "That CSV has headers but no data rows, so the current project was not changed.")
        return False

    type_error = translation_data_type_error(dataframe)
    if type_error:
        add_message(document, "user", f"Uploaded translation data: **{uploaded.name}**")
        add_message(document, "assistant", type_error)
        return False

    upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if upload_hash == document.get("translation_upload_hash"):
        add_message(document, "user", f"Uploaded translation data: **{uploaded.name}**")
        add_message(document, "assistant", "This is already the active translation-data file. No project data changed.")
        return True

    profiles = profile_columns(dataframe)
    candidates = [profile.name for profile in profiles if profile.selected]
    summary = ", ".join(f"`{name}`" for name in candidates) or "none"
    game_mode = detect_game_schema(dataframe)
    game_config = infer_game_config(dataframe) if game_mode else None

    existing_bible = document.get("character_bible")
    bible_is_configured = document.get("bible_revision", 0) > 0
    if game_mode and bible_is_configured and existing_bible is not None and not existing_bible.empty:
        refreshed_bible = existing_bible.copy().fillna("")
        defaults = default_character_bible(dataframe, game_config).fillna("")
        for column in defaults.columns:
            if column not in refreshed_bible.columns:
                refreshed_bible[column] = ""
        known_speakers = set(refreshed_bible.get("speaker", pd.Series(dtype=str)).astype(str).str.strip())
        missing_rows = defaults[~defaults["speaker"].astype(str).str.strip().isin(known_speakers)]
        if not missing_rows.empty:
            refreshed_bible = pd.concat(
                [refreshed_bible[list(defaults.columns)], missing_rows[list(defaults.columns)]],
                ignore_index=True,
            )
        else:
            refreshed_bible = refreshed_bible[list(defaults.columns)]
    elif game_mode:
        refreshed_bible = default_character_bible(dataframe, game_config)
    else:
        refreshed_bible = existing_bible if bible_is_configured and existing_bible is not None else pd.DataFrame()

    document.update({
        "name": uploaded.name,
        "size": uploaded.size,
        "translation_upload_hash": upload_hash,
        "dataframe": dataframe,
        "profiles": profiles,
        "game_mode": game_mode,
        "game_config": asdict(game_config) if game_config else {},
        "character_bible": refreshed_bible,
        "result": None,
        "review_table": None,
        "job_id": None,
        "last_job_error": "",
        "plan_preview": None,
        "plan_preview_error": "",
        "send_bible_warning": "",
        "awaiting_bible_override": False,
        "failure_triage": None,
        "failure_triage_signature": "",
        "failure_triage_error": "",
        "failure_triage_notice": "",
        "result_glossary_revision": 0,
        "result_bible_revision": 0,
        "result_target_languages": [],
        "glossary_rerun_ready": False,
        "pending_full_rerun": False,
        "pending_failed_retry": False,
        "pending_game_rerun": None,
    })
    touch_workspace_item(document, "translation")
    add_message(document, "user", f"Updated translation data: **{uploaded.name}**")
    add_message(
        document,
        "assistant",
        f"I replaced this project's source package with **{len(dataframe):,} rows × "
        f"{len(dataframe.columns)} columns**. Suggested text columns: {summary}. "
        "The existing Character Bible, glossary, target languages, instructions, and run history were preserved. "
        "The previous translation result was cleared because it belonged to the earlier source file.",
    )
    st.session_state.active_document_id = document_id
    st.session_state.uploader_version += 1
    ensure_game_status_message(document)
    return True


def handle_onboarding_bible_toolbar_upload(upload_key: str) -> None:
    uploaded = st.session_state.get(upload_key)
    if uploaded is None:
        return
    imported, error = read_uploaded_character_bible(uploaded)
    st.session_state.onboarding_bible_error = error
    if imported is None:
        st.session_state.onboarding_messages = [
            *st.session_state.get("onboarding_messages", []),
            {"role": "user", "content": f"Uploaded Character Bible: **{uploaded.name}**"},
            {"role": "assistant", "content": error},
        ]
        return
    bible_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if bible_hash == st.session_state.get("pending_character_bible_hash"):
        return
    st.session_state.pending_character_bible = imported
    st.session_state.pending_character_bible_name = uploaded.name
    st.session_state.pending_character_bible_hash = bible_hash
    touch_pending_workspace_item("bible")
    st.session_state.onboarding_messages = [
        *st.session_state.get("onboarding_messages", []),
        {"role": "user", "content": f"Uploaded Character Bible: **{uploaded.name}**"},
        {"role": "assistant", "content": "Character Bible validated and saved."},
    ]
    st.session_state.onboarding_action = ""


def handle_project_bible_toolbar_upload(document_id: str, upload_key: str) -> None:
    uploaded = st.session_state.get(upload_key)
    if uploaded is None:
        return
    document = st.session_state.documents[document_id]
    imported, error = read_uploaded_character_bible(uploaded)
    document["bible_upload_error"] = error
    if imported is None:
        add_message(document, "user", f"Uploaded Character Bible: **{uploaded.name}**")
        add_message(document, "assistant", error)
        return
    bible_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if bible_hash == document.get("bible_upload_hash"):
        return
    current = document.get("character_bible")
    if current is None or "speaker" not in current.columns:
        current = default_character_bible(
            document["dataframe"], GameConfig(**document["game_config"])
        )
    for column in current.columns:
        if column not in imported:
            imported[column] = ""
    document["character_bible"] = imported[list(current.columns)].fillna("")
    document["character_bible_name"] = uploaded.name
    document["bible_upload_hash"] = bible_hash
    document["bible_revision"] = document.get("bible_revision", 0) + 1
    touch_workspace_item(document, "bible")
    document["send_bible_warning"] = ""
    document["awaiting_bible_override"] = False
    add_message(document, "user", f"Uploaded Character Bible: **{uploaded.name}**")
    add_message(document, "assistant", "Character Bible validated and saved.")
    document["conversation_action"] = ""
    ensure_game_status_message(document)


def send_onboarding_message(message: str | None = None) -> None:
    """Keep pre-upload instructions as part of the new project's conversation."""
    message = str(
        message if message is not None else st.session_state.get("onboarding_draft") or ""
    ).strip()
    if not message:
        st.session_state.onboarding_empty_message = True
        return
    st.session_state.onboarding_empty_message = False
    st.session_state.onboarding_instructions = message
    st.session_state.onboarding_messages = [
        *st.session_state.get("onboarding_messages", []),
        {"role": "user", "content": message},
        {
            "role": "assistant",
            "content": "Instructions saved. Upload either file first; I’ll carry this guidance into the project.",
        },
    ]


def render_onboarding_bible_upload() -> None:
    bible_upload = st.file_uploader(
        "Upload Character Bible CSV",
        type=["csv"],
        key=f"onboarding_bible_{st.session_state.uploader_version}",
    )
    if bible_upload is None:
        if st.session_state.get("pending_character_bible") is not None:
            st.success(f"Character Bible ready: {st.session_state.pending_character_bible_name}")
        return
    bible_hash = hashlib.sha256(bible_upload.getvalue()).hexdigest()
    if bible_hash == st.session_state.get("pending_character_bible_hash"):
        st.success(f"Character Bible ready: {bible_upload.name}")
        return
    try:
        imported = read_csv(bible_upload)
    except Exception as error:
        st.error(f"I couldn’t read this Character Bible CSV. Details: {error}")
        return
    errors = validate_character_bible(imported)
    if errors:
        st.error("This file does not look like a Character Bible.\n\n- " + "\n- ".join(errors))
        return
    st.session_state.pending_character_bible = imported.fillna("")
    st.session_state.pending_character_bible_name = bible_upload.name
    st.session_state.pending_character_bible_hash = bible_hash
    touch_pending_workspace_item("bible")
    st.session_state.onboarding_messages = [
        *st.session_state.get("onboarding_messages", []),
        {"role": "user", "content": f"Uploaded Character Bible: **{bible_upload.name}**"},
        {
            "role": "assistant",
            "content": "Character Bible validated and saved. You can upload the translation data next.",
        },
    ]
    st.success(f"Character Bible ready: {bible_upload.name}. Now upload translation data.")


def render_onboarding_action_tray() -> None:
    """Keep direct upload and setup controls in one stable bottom toolbar."""
    upload_action, bible_action = st.columns(2)
    with upload_action, st.container(key="translation_upload_action"):
        st.button(
            "Upload translation data",
            key="open_onboarding_translation_upload",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
            help="Choose a CSV to create its conversation and workspace.",
            on_click=open_onboarding_action,
            args=("translation_upload",),
        )
    with bible_action, st.container(key="bible_upload_action"):
        pending_bible_ready = st.session_state.get("pending_character_bible") is not None
        st.button(
            "Update Bible" if pending_bible_ready else "Upload Character Bible",
            key="open_onboarding_bible_upload",
            type=(
                "secondary"
                if pending_bible_ready
                else "primary"
            ),
            icon=":material/menu_book:",
            width="stretch",
            help="Add speaker voice, tone, pronoun, and relationship guidance.",
            on_click=open_onboarding_action,
            args=("bible_upload",),
        )
    language_action, glossary_action = st.columns(2)
    with language_action:
        st.button(
            "Target languages ✓"
            if st.session_state.get("onboarding_target_languages")
            else "Target languages",
            key="open_onboarding_languages",
            type=(
                "secondary"
                if st.session_state.get("onboarding_target_languages")
                else "primary"
            ),
            icon=":material/language:",
            width="stretch",
            on_click=open_onboarding_language_action,
        )
    with glossary_action:
        pending_glossary_ready = bool(
            str(st.session_state.get("onboarding_glossary_text", "")).strip()
        )
        st.button(
            "Update glossary" if pending_glossary_ready else "Glossary",
            key="open_onboarding_glossary",
            type="secondary",
            icon=":material/book_2:",
            width="stretch",
            on_click=open_onboarding_action,
            args=("glossary",),
        )


def create_uploaded_project(
    uploaded, project_name: str = "", conversation_lead: str = ""
) -> bool:
    """Create or reopen one independent project from a Conversation upload."""
    raw = uploaded.getvalue()
    document_id = hashlib.sha256(raw).hexdigest()[:16]
    if document_id not in st.session_state.documents:
        try:
            dataframe = read_csv(uploaded)
        except Exception as error:
            st.error(f"I couldn’t read that CSV: {error}")
            return False
        if dataframe.empty:
            st.error("The CSV has headers but no data rows.")
            return False
        type_error = translation_data_type_error(dataframe)
        if type_error:
            st.session_state.onboarding_messages = [
                *st.session_state.get("onboarding_messages", []),
                {"role": "user", "content": f"Uploaded translation data: **{uploaded.name}**"},
                {"role": "assistant", "content": type_error},
            ]
            st.session_state.onboarding_translation_error = type_error
            return False
        profiles = profile_columns(dataframe)
        candidates = [profile.name for profile in profiles if profile.selected]
        summary = ", ".join(f"`{name}`" for name in candidates) or "none"
        game_mode = detect_game_schema(dataframe)
        game_config = infer_game_config(dataframe) if game_mode else None
        pending_glossary_text = str(st.session_state.get("onboarding_glossary_text", ""))
        pending_glossary_entries, _ = parse_glossary(pending_glossary_text)
        onboarding_messages = list(st.session_state.get("onboarding_messages", []))
        if conversation_lead and not any(
            message.get("content") == conversation_lead
            for message in onboarding_messages
        ):
            onboarding_messages.append({"role": "assistant", "content": conversation_lead})
        document = {
            "id": document_id,
            "name": uploaded.name,
            "project_name": project_name.strip() or project_name_from_file(uploaded.name),
            "size": uploaded.size,
            "translation_upload_hash": hashlib.sha256(raw).hexdigest(),
            "dataframe": dataframe,
            "profiles": profiles,
            "messages": [{
                "role": "assistant",
                "content": (
                    "Hi! I’m your localization agent. Start by uploading translation data or a "
                    "Character Bible—you can upload either one first."
                ),
            }] + onboarding_messages,
            "result": None,
            "unread": False,
            "job_id": None,
            "last_job_error": "",
            "draft": "",
            "translation_instructions": str(st.session_state.get("onboarding_instructions", "")),
            "target_languages": list(st.session_state.get("onboarding_target_languages", [])),
            "glossary_text": pending_glossary_text,
            "glossary_draft": pending_glossary_text,
            "glossary_validation_errors": [],
            "glossary_confirmation": "",
            "runs": [],
            "quality_feedback": {},
            "quality_version": 0,
            "empty_prompt_error": False,
            "applied_glossary_signature": glossary_signature(pending_glossary_entries),
            "glossary_revision": 1 if pending_glossary_entries else 0,
            "bible_revision": 0,
            "character_bible_name": "",
            "workspace_item_order": [
                *reversed(ordered_pending_workspace_items()),
                "translation",
            ],
            "result_glossary_revision": 0,
            "result_bible_revision": 0,
            "result_target_languages": [],
            "glossary_dirty": False,
            "glossary_rerun_ready": False,
            "glossary_ui_version": 2,
            "game_mode": game_mode,
            "game_config": asdict(game_config) if game_config else {},
            "character_bible": (
                default_character_bible(dataframe, game_config)
                if game_config else pd.DataFrame()
            ),
            "review_table": None,
            "review_version": 0,
            "evaluate_style": False,
            "plan_preview": None,
            "plan_preview_error": "",
            "send_bible_warning": "",
            "awaiting_bible_override": False,
            "failure_triage": None,
            "failure_triage_signature": "",
            "failure_triage_version": 0,
            "failure_triage_glossary_revision": 0,
            "failure_triage_bible_revision": 0,
            "failure_triage_error": "",
            "failure_triage_notice": "",
            "failure_resolutions": [],
            "conversation_action": "",
        }
        pending_bible = st.session_state.get("pending_character_bible")
        if game_config and pending_bible is not None:
            current_bible = document["character_bible"]
            pending_bible = pending_bible.copy()
            for column in current_bible.columns:
                if column not in pending_bible:
                    pending_bible[column] = ""
            document["character_bible"] = pending_bible[list(current_bible.columns)].fillna("")
            document["bible_revision"] = 1
            document["character_bible_name"] = str(
                st.session_state.get("pending_character_bible_name", "")
            )
        add_message(document, "user", f"Uploaded translation data: **{uploaded.name}**")
        add_message(
            document,
            "assistant",
            f"I inspected **{len(dataframe):,} rows × {len(dataframe.columns)} columns**. "
            f"Suggested Chinese text columns: {summary}. "
            + (
                "I also detected a game-dialogue schema, so character-aware localization mode is ready."
                if game_mode else "This file now has its own independent workspace."
            ),
        )
        st.session_state.documents[document_id] = document
    else:
        document = st.session_state.documents[document_id]
        if conversation_lead and (
            not document.get("messages")
            or document["messages"][-1].get("content") != conversation_lead
        ):
            add_message(document, "assistant", conversation_lead)
        add_message(document, "user", f"Uploaded translation data: **{uploaded.name}**")
        add_message(
            document,
            "assistant",
            "Opened the existing project for this file. Its conversation and results were preserved.",
        )
    save_active_draft()
    st.session_state.active_document_id = document_id
    st.session_state.documents[document_id]["unread"] = False
    st.session_state.uploader_version += 1
    st.session_state.onboarding_panel = ""
    st.session_state.onboarding_messages = []
    st.session_state.onboarding_draft = ""
    st.session_state.onboarding_instructions = ""
    st.session_state.onboarding_target_languages = []
    st.session_state.onboarding_action = ""
    st.session_state.onboarding_glossary_text = ""
    st.session_state.pending_character_bible = None
    st.session_state.pending_character_bible_name = ""
    st.session_state.pending_character_bible_hash = ""
    st.session_state.pending_workspace_item_order = []
    st.session_state.pending_bible_editor_version = 0
    return True


def main() -> None:
    initialize_state()
    collect_completed_jobs()
    st.title("🌐 Multilingual Translation Agent")
    st.caption("Deterministic CSV handling · LLM translation · measurable validation")
    st.markdown(
        """
        <style>
        .st-key-toolbar_dialogue_upload [data-testid="stFileUploaderDropzone"]
        [data-testid="stBaseButton-secondary"],
        .st-key-toolbar_bible_upload [data-testid="stFileUploaderDropzone"]
        [data-testid="stBaseButton-secondary"] {
            background: #2563EB !important;
            color: white !important;
            border-color: #2563EB !important;
            width: 100% !important;
        }
        .st-key-toolbar_dialogue_upload [data-testid="stFileUploaderDropzoneInstructions"],
        .st-key-toolbar_bible_upload [data-testid="stFileUploaderDropzoneInstructions"],
        .st-key-glossary_upload_compact [data-testid="stFileUploaderDropzoneInstructions"],
        .st-key-glossary_upload_compact small {
            display: none !important;
        }
        .st-key-toolbar_dialogue_upload [data-testid="stFileUploaderDropzone"],
        .st-key-toolbar_bible_upload [data-testid="stFileUploaderDropzone"],
        .st-key-glossary_upload_compact [data-testid="stFileUploaderDropzone"] {
            min-height: 2.5rem !important;
            padding: 0 !important;
            border: 0 !important;
            background: transparent !important;
        }
        /* Guided uploads are temporary chat actions. Give the entire newest
           message a blue treatment so the file picker cannot be overlooked. */
        [data-testid="stChatMessage"]:has([data-testid="stFileUploader"])
        [data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed #3B82F6 !important;
            background: #DBEAFE !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stFileUploader"])
        [data-testid="stFileUploaderDropzone"] button {
            border-color: #2563EB !important;
            background: #2563EB !important;
            color: #FFFFFF !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stFileUploader"])
        [data-testid="stFileUploaderDropzoneInstructions"] {
            color: #1E3A8A !important;
        }
        /* st.bottom spans the whole page by default, so a taller Conversation
           toolbar also reserves/paints empty space over Workspace. Keep the
           bottom layer out of document flow and make only its left column live. */
        [data-testid="stBottom"] {
            position: fixed !important;
            bottom: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stAppViewContainer"]:has(
            > [data-testid="stSidebar"][aria-expanded="true"]
        ) [data-testid="stBottom"] {
            left: 300px !important;
            width: calc(100vw - 300px) !important;
        }
        [data-testid="stAppViewContainer"]:has(
            > [data-testid="stSidebar"][aria-expanded="false"]
        ) [data-testid="stBottom"] {
            left: 0 !important;
            width: 100vw !important;
        }
        [data-testid="stBottom"] > div {
            background: transparent !important;
        }
        [data-testid="stBottomBlockContainer"] {
            padding-top: 0 !important;
            padding-bottom: 1rem !important;
            pointer-events: none !important;
        }
        [data-testid="stBottomBlockContainer"]
        > [data-testid="stVerticalBlock"]
        > [data-testid="stLayoutWrapper"]
        > [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"]:first-child {
            pointer-events: auto !important;
            background: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 0.75rem !important;
            box-shadow: 0 -8px 24px rgba(15, 23, 42, 0.08) !important;
            padding: 0.75rem !important;
        }
        @media (max-width: 768px) {
            [data-testid="stBottom"] {
                left: 0 !important;
                width: 100vw !important;
            }
        }
        .st-key-glossary_composer [data-testid="stChatInputSubmitButton"] {
            width: 7.75rem !important;
            border-radius: 0.65rem !important;
        }
        .st-key-glossary_composer [data-testid="stChatInputSubmitButton"] svg {
            display: none !important;
        }
        .st-key-glossary_composer [data-testid="stChatInputSubmitButton"]::after {
            content: "Apply glossary";
            white-space: nowrap;
            font-size: 0.875rem;
            font-weight: 600;
        }
        .st-key-translation_upload_action button,
        .st-key-bible_upload_action button {
            padding-left: 0.35rem !important;
            padding-right: 0.35rem !important;
        }
        .st-key-translation_upload_action button p,
        .st-key-bible_upload_action button p {
            font-size: 0.72rem !important;
            letter-spacing: -0.01em !important;
        }
        [data-testid="stBottom"] [data-testid="stButton"] button,
        [data-testid="stBottom"] [data-testid="stDownloadButton"] button {
            min-height: 2.25rem !important;
            padding: 0.35rem 0.3rem !important;
        }
        [data-testid="stBottom"] [data-testid="stButton"] button p,
        [data-testid="stBottom"] [data-testid="stDownloadButton"] button p {
            font-size: 0.72rem !important;
            letter-spacing: -0.015em !important;
            white-space: nowrap !important;
        }
        .st-key-review_header_download [data-testid="stDownloadButton"] {
            display: flex;
            justify-content: flex-end;
        }
        .st-key-review_header_download button p {
            white-space: nowrap;
        }
        .st-key-conversation_history [data-testid="stChatMessage"] {
            padding-top: 0.35rem !important;
            padding-bottom: 0.35rem !important;
            margin-bottom: 0.45rem !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 0.75rem !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
            overflow: hidden !important;
        }
        .st-key-conversation_history {
            padding: 0.65rem !important;
            border-radius: 0.75rem !important;
            background: #F3F4F6 !important;
            height: var(--conversation-available-height, max(11rem, calc(100dvh - 27.8rem))) !important;
            min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 27.8rem))) !important;
            max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 27.8rem))) !important;
            overflow-y: auto !important;
            overscroll-behavior: contain !important;
            scrollbar-gutter: stable;
        }
        .st-key-conversation_history > :first-child {
            margin-top: 0 !important;
        }
        [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_history) {
            height: var(--conversation-available-height, max(11rem, calc(100dvh - 27.8rem))) !important;
            min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 27.8rem))) !important;
            max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 27.8rem))) !important;
            overflow: hidden !important;
        }
        body:has(.st-key-demo_mode_banner) .st-key-conversation_history {
            height: var(--conversation-available-height, max(11rem, calc(100dvh - 31.25rem))) !important;
            min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 31.25rem))) !important;
            max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 31.25rem))) !important;
        }
        body:has(.st-key-demo_mode_banner)
        [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_history) {
            height: var(--conversation-available-height, max(11rem, calc(100dvh - 31.25rem))) !important;
            min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 31.25rem))) !important;
            max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 31.25rem))) !important;
        }
        @media (min-width: 1500px) {
            .st-key-conversation_history,
            [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_history) {
                height: var(--conversation-available-height, max(11rem, calc(100dvh - 19.05rem))) !important;
                min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 19.05rem))) !important;
                max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 19.05rem))) !important;
            }
            body:has(.st-key-demo_mode_banner) .st-key-conversation_history,
            body:has(.st-key-demo_mode_banner)
            [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_history) {
                height: var(--conversation-available-height, max(11rem, calc(100dvh - 22.5rem))) !important;
                min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 22.5rem))) !important;
                max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 22.5rem))) !important;
            }
            body:has(.st-key-demo_mode_banner):has(.st-key-active_sidebar_project)
            .st-key-conversation_history,
            body:has(.st-key-demo_mode_banner):has(.st-key-active_sidebar_project)
            [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_history) {
                height: var(--conversation-available-height, max(11rem, calc(100dvh - 24.3rem))) !important;
                min-height: var(--conversation-available-height, max(11rem, calc(100dvh - 24.3rem))) !important;
                max-height: var(--conversation-available-height, max(11rem, calc(100dvh - 24.3rem))) !important;
            }
        }
        .st-key-onboarding_composer {
            position: sticky;
            bottom: 0;
            z-index: 10;
            background: var(--st-background-color);
            border: 1px solid var(--st-border-color);
            border-radius: var(--st-base-radius);
            padding: 0.75rem;
            box-shadow: 0 -8px 24px rgba(15, 23, 42, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if demo_mode_enabled():
        with st.container(key="demo_mode_banner"):
            st.warning(
                "Demo Mode is ON: translations are synthetic, `[FAIL]` fails on the first run and succeeds on targeted retry, "
                "and no OpenAI API calls are made."
            )

    documents = st.session_state.documents
    with st.sidebar:
        st.header("Game projects")
        st.button(
            "＋ New dialogue",
            key="new_dialogue_project",
            type="secondary",
            width="stretch",
            on_click=open_new_project_upload,
        )
        if documents:
            st.caption(f"{len(documents)} open project{'s' if len(documents) != 1 else ''}")
            for item_id, item in documents.items():
                job_status = background_jobs().status(item.get("job_id"))
                unread = item.get("unread", False)
                prefix = "⏳ " if job_status["state"] in {"running", "cancelling"} else "🔵 " if unread else ""
                label = f"{prefix}{item.get('project_name', project_name_from_file(item['name']))}"
                project_container = st.container(
                    key=(
                        "active_sidebar_project"
                        if item_id == st.session_state.active_document_id
                        else f"sidebar_project_{item_id}"
                    )
                )
                with project_container:
                    st.button(
                        label,
                        key=f"open_document_{item_id}",
                        type="secondary",
                        width="stretch",
                        on_click=switch_document,
                        args=(item_id,),
                    )
            if any(background_jobs().status(item.get("job_id"))["state"] in {"running", "cancelling"} for item in documents.values()):
                watch_background_jobs()
        else:
            st.caption("Upload a dialogue CSV from Conversation to begin.")

    active_id = st.session_state.active_document_id
    if not active_id or active_id not in documents:
        conversation, workspace = st.columns([1, 1.4], gap="large")
        with conversation:
            with st.container(key="conversation_history", height=360, border=False):
                with st.chat_message("assistant"):
                    st.markdown(
                        "Hi! I’m your localization agent. Start by uploading translation data or a Character Bible—you can upload either one first."
                    )
                for message in st.session_state.get("onboarding_messages", []):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                render_onboarding_guided_action()
        with workspace:
            pending_workspace_items = ordered_pending_workspace_items()
            if not pending_workspace_items:
                st.subheader("Workspace")
                st.caption("Your file summary, translation progress, preview, and results will appear here.")
            for workspace_item in pending_workspace_items:
                if workspace_item == "bible":
                    render_pending_character_bible_summary()
                elif workspace_item == "glossary":
                    render_pending_glossary_summary()
        with st.bottom:
            composer_column, _ = st.columns([1, 1.4], gap="large")
            with composer_column:
                render_onboarding_action_tray()
                glossary_mode = st.session_state.get("onboarding_action") == "glossary"
                with st.container(key="glossary_composer" if glossary_mode else "instruction_composer"):
                    onboarding_message = st.chat_input(
                        "Terminology rules…" if glossary_mode else "Optional instructions…",
                        key=(
                            "onboarding_glossary_chat_input"
                            if glossary_mode else "onboarding_chat_input"
                        ),
                    )
                if onboarding_message:
                    if glossary_mode:
                        apply_onboarding_chat_glossary(onboarding_message)
                    else:
                        send_onboarding_message(onboarding_message)
                    st.rerun()
        keep_conversation_scrolled_to_latest()
        return

    document = documents[active_id]
    dataframe = document["dataframe"]
    draft_key = f"draft_input_{active_id}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = document.get("draft", "")
    st.markdown(
        """
        <style>
        .st-key-persistent_result_panel {
            position: relative;
            top: auto;
            box-sizing: border-box;
            height: calc(100vh - 23rem);
            min-height: 10rem;
            max-height: calc(100vh - 23rem);
            overflow-y: auto;
            padding-right: 0.35rem;
            padding-bottom: 1rem;
            overscroll-behavior: contain;
        }
        @media (min-width: 769px) {
            [data-testid="stAppScrollToBottomContainer"] {
                overflow-y: hidden !important;
            }
        }
        .st-key-chat_composer {
            position: sticky;
            bottom: 0;
            z-index: 10;
            background: var(--st-background-color);
            border: 1px solid var(--st-border-color);
            border-radius: var(--st-base-radius);
            padding: 0.75rem;
            box-shadow: 0 -8px 24px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stChatMessage"] {
            border-radius: var(--st-base-radius);
            padding: 0.35rem 0.5rem;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            justify-content: flex-start;
            text-align: left;
        }
        [data-testid="stSidebar"] .st-key-active_sidebar_project
        [data-testid="stButton"] button {
            background: color-mix(
                in srgb,
                var(--st-primary-color) 20%,
                transparent
            ) !important;
            border-color: color-mix(
                in srgb,
                var(--st-primary-color) 38%,
                var(--st-border-color)
            ) !important;
            color: var(--st-text-color) !important;
        }
        [data-testid="stDialog"] {
            background: rgba(15, 23, 42, 0.45) !important;
        }
        [data-testid="stDialog"][role="dialog"],
        [data-testid="stDialog"] [role="dialog"],
        [data-testid="stModal"] [role="dialog"] {
            position: fixed !important;
            top: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            left: 0 !important;
            margin: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
            height: 100vh !important;
            max-height: 100vh !important;
            border-radius: 0 !important;
            overflow-y: auto !important;
            background: #ffffff !important;
            color: #111827 !important;
            --st-background-color: #ffffff;
            --st-secondary-background-color: #f5f7fa;
            --st-text-color: #111827;
        }
        [data-testid="stDialog"] [role="dialog"] > div,
        [data-testid="stModal"] [role="dialog"] > div {
            background: #ffffff !important;
        }
        .st-key-review_directory_panel {
            width: 100%;
            height: auto;
            max-height: none;
            overflow: visible;
            background: transparent;
            border: 0;
            padding: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    conversation, results_panel = st.columns([1, 1.4], gap="large")

    with results_panel:
        with st.container(key="persistent_result_panel"):
            if document.get("last_job_error"):
                st.error(document["last_job_error"], icon="🚨")

            profiles = document["profiles"]
            proposed = [profile.name for profile in profiles if profile.selected]
            if document.get("game_mode") and document.get("game_config"):
                selected_columns = [GameConfig(**document["game_config"]).source_text]
            else:
                selected_columns = list(document.get("selected_columns") or proposed)

            for workspace_item in ordered_workspace_items(document):
                if workspace_item == "translation":
                    render_active_file_summary(active_id)
                elif workspace_item == "bible":
                    render_character_bible_summary(active_id)
                elif workspace_item == "glossary":
                    render_glossary_summary(active_id)
                elif workspace_item == "result":
                    render_result(document["result"], document)
                    render_post_setup_result(document["result"], document)

    pending_game_rerun = document.pop("pending_game_rerun", None)
    if pending_game_rerun:
        result = document.get("result")
        review_table = document.get("review_table")
        current_glossary, _ = parse_glossary(document.get("glossary_text", ""))
        rows = pending_game_rerun.get("row_positions")
        if rows is None:
            rows = select_review_rows(
                review_table,
                pending_game_rerun["scope"],
                pending_game_rerun.get("value", ""),
                [entry.source for entry in current_glossary],
            )
        if not rows:
            add_message(document, "assistant", "No game dialogue lines match that rerun scope.")
            st.rerun()
        glossary_entries, glossary_errors = parse_glossary(document.get("glossary_text", ""))
        if glossary_errors:
            add_message(document, "assistant", "Fix the glossary format before rerunning: " + "; ".join(glossary_errors))
            st.rerun()
        key = api_key()
        if not key:
            add_message(document, "assistant", "The app is ready, but OPENAI_API_KEY has not been configured.")
            st.rerun()
        add_message(
            document,
            "assistant",
            f"Rerunning {len(rows)} game dialogue line(s) from scope: {pending_game_rerun['scope']}. Reviewed and kept-source lines stay locked.",
        )
        start_background_job(document,
            dataframe.copy(deep=True),
            result.source_columns,
            result.target_languages,
            key,
            glossary=glossary_entries,
            glossary_revision=document.get("glossary_revision", 0),
            game_config=GameConfig(**document["game_config"]),
            character_bible=document["character_bible"].copy(deep=True),
            row_positions=rows,
            game_base_result=result,
            game_approved_locks=approved_locks(review_table),
            evaluate_style=document.get("evaluate_style", False),
        )
        st.rerun()

    if document.pop("pending_full_rerun", False):
        if document.get("job_id"):
            add_message(document, "assistant", "A translation is already running for this file.")
            st.rerun()
        result = document.get("result")
        if result is None:
            add_message(document, "assistant", "There is no completed translation configuration to rerun yet.")
            st.rerun()
        glossary_entries, glossary_errors = parse_glossary(document.get("glossary_text", ""))
        if glossary_errors:
            add_message(document, "assistant", "Fix the glossary format before rerunning: " + "; ".join(glossary_errors))
            st.rerun()
        key = api_key()
        if not key:
            add_message(document, "assistant", "The app is ready, but `OPENAI_API_KEY` has not been configured by the deployer.")
            st.rerun()
        changed_inputs = document.get("pending_rerun_updates", [])
        change_summary = ", ".join(changed_inputs) if changed_inputs else "current settings"
        add_message(
            document,
            "assistant",
            f"Rerunning the full translation with {change_summary}.",
        )
        game_kwargs = {}
        if result.game_config:
            game_kwargs = {
                "game_config": GameConfig(**document["game_config"]),
                "character_bible": document["character_bible"].copy(deep=True),
                "game_base_result": result,
                "game_approved_locks": approved_locks(document.get("review_table")),
                "evaluate_style": document.get("evaluate_style", False),
            }
        start_background_job(document,
            dataframe.copy(deep=True),
            result.source_columns,
            list(document.get("target_languages", [])) or result.target_languages,
            key,
            glossary=glossary_entries,
            glossary_revision=document.get("glossary_revision", 0),
            **game_kwargs,
        )
        st.rerun()

    if document.pop("pending_failed_retry", False):
        if document.get("job_id"):
            add_message(document, "assistant", "A translation is already running for this file.")
            st.rerun()
        result = document.get("result")
        if result is None or result.failures.empty:
            add_message(document, "assistant", "There are no failed values to retry.")
            st.rerun()
        glossary_entries, glossary_errors = parse_glossary(document.get("glossary_text", ""))
        if glossary_errors:
            add_message(document, "assistant", "Fix the glossary format before retrying: " + "; ".join(glossary_errors))
            st.rerun()
        key = api_key()
        if not key:
            add_message(document, "assistant", "The app is ready, but `OPENAI_API_KEY` has not been configured by the deployer.")
            st.rerun()
        add_message(document, "assistant", f"Retrying only **{len(result.failures)} failed unique value(s)** in the background.")
        if result.game_config:
            start_background_job(document,
                dataframe.copy(deep=True),
                result.source_columns,
                result.target_languages,
                key,
                glossary=glossary_entries,
                glossary_revision=document.get("glossary_revision", 0),
                game_config=GameConfig(**document["game_config"]),
                character_bible=document["character_bible"].copy(deep=True),
                row_positions=sorted(set(result.failures["row_position"].astype(int))),
                game_base_result=result,
                game_approved_locks=approved_locks(document.get("review_table")),
                evaluate_style=document.get("evaluate_style", False),
                game_retry_failures=True,
            )
        else:
            start_background_job(document,
                dataframe.copy(deep=True),
                result.source_columns,
                result.target_languages,
                key,
                glossary=glossary_entries,
                previous_result=result,
                glossary_revision=document.get("glossary_revision", 0),
            )
        st.rerun()

    with conversation:
        ensure_game_status_message(document)
        ensure_no_failed_values_message(document)
        document["unread"] = False
        active_job = background_jobs().status(document.get("job_id"))
        with st.container(key="conversation_history", height=360, border=False):
            for message in document["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            render_conversation_guided_action(document)

            if active_job["state"] in {"running", "cancelling"}:
                metadata = active_job.get("metadata", {})
                if active_job["total"]:
                    st.info(
                        f"{active_job['label']} · "
                        f"batch {active_job['done']}/{active_job['total']}. You can open another file."
                    )
                else:
                    st.info("Translation is starting in the background. You can open another file.")
                st.caption(
                    f"Estimate: {metadata.get('estimated_values', 0):,} unique values · "
                    f"~{metadata.get('estimated_input_tokens', 0) + metadata.get('estimated_output_tokens', 0):,} tokens · "
                    f"~${metadata.get('estimated_cost_usd', 0):.5f} USD using {metadata.get('model', 'configured model')}. "
                    "Actual provider billing may differ."
                )
                st.button(
                    "Cancel translation" if active_job["state"] == "running" else "Cancelling…",
                    key=f"cancel_job_{active_id}",
                    disabled=active_job["state"] == "cancelling",
                    on_click=cancel_translation,
                    args=(active_id,),
                )

        prompt = document.pop("pending_prompt", None)
        if prompt:
            add_message(document, "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            languages = parse_languages(prompt)
            if not languages:
                add_message(
                    document,
                    "assistant",
                    "I couldn’t identify a supported target language. Try English, Japanese, Korean, French, German, Spanish, Portuguese, Italian, Traditional Chinese, or Simplified Chinese.",
                )
                st.rerun()
            if not selected_columns:
                add_message(document, "assistant", "No source columns are selected. Choose at least one column in “Review detected columns.”")
                st.rerun()
            glossary_entries, glossary_errors = parse_glossary(document.get("glossary_text", ""))
            if glossary_errors:
                add_message(document, "assistant", "Fix the glossary format before translating: " + "; ".join(glossary_errors))
                st.rerun()
            key = api_key()
            if not key:
                add_message(document, "assistant", "The app is ready, but `OPENAI_API_KEY` has not been configured by the deployer.")
                st.rerun()
            game_kwargs = {}
            if document.get("game_mode"):
                game_kwargs = {
                    "game_config": GameConfig(**document["game_config"]),
                    "character_bible": document["character_bible"].copy(deep=True),
                    "evaluate_style": document.get("evaluate_style", False),
                }
            start_background_job(document,
                dataframe.copy(deep=True),
                list(selected_columns),
                languages,
                key,
                glossary=glossary_entries,
                glossary_revision=document.get("glossary_revision", 0),
                **game_kwargs,
            )
            st.rerun()

    with st.bottom:
        composer_column, _ = st.columns([1, 1.4], gap="large")
        with composer_column:
            render_conversation_action_tray(document, active_job)
            glossary_mode = document.get("conversation_action") == "glossary"
            with st.container(key="glossary_composer" if glossary_mode else "instruction_composer"):
                optional_instruction = st.chat_input(
                    "Terminology rules…" if glossary_mode else "Optional instructions…",
                    key=(
                        f"glossary_chat_input_{active_id}"
                        if glossary_mode else f"chat_input_{active_id}"
                    ),
                )
            if optional_instruction:
                if glossary_mode:
                    apply_chat_glossary(active_id, optional_instruction)
                else:
                    send_optional_instruction(active_id, optional_instruction)
                st.rerun()
    keep_conversation_scrolled_to_latest()


if __name__ == "__main__":
    main()
