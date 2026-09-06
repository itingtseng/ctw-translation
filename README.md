# Multilingual Translation Agent

A deployable, chat-based CSV translation workflow for Chinese source data. Upload a CSV, review the agent's proposed text columns, choose one or more target languages from the Conversation multi-select (or type a natural-language request), preview the validated result, and download an Excel-friendly UTF-8 CSV.

The implementation is deliberately lean: Streamlit for interaction, pandas for deterministic data handling, and one LLM API for the probabilistic translation step. There is no RAG pipeline or vector database because this task neither retrieves knowledge nor needs semantic search.

## Quick start (2 minutes)

1. Open the app.
2. Upload [`samples/interview/large_context_localization_demo.csv`](samples/interview/large_context_localization_demo.csv) as the translation data, then upload its matching [`samples/interview/large_context_character_bible_demo.csv`](samples/interview/large_context_character_bible_demo.csv) as the Character Bible. This is the recommended interview demo: 120 Chinese dialogue rows with linked scenes, character context, and deliberate context-review cases.
3. Type `Translate to English and Japanese` in the chat.
4. Preview the result, then download the translated CSV.

Everything below this point is the detailed design doc: architecture, reliability/fallback behavior, model routing, evaluation, and trade-offs. It's here for anyone reviewing the engineering, not required reading to try the app.

## Product flow

1. The empty state is already a complete Conversation: a greeting offers primary actions for uploading translation data or a Character Bible in either order, Target languages are available immediately, and the fixed composer can save instructions before any file exists. Upload and validation events become chronological chat messages instead of disappearing when the dialogue project is created.
2. The agent inspects the schema and samples up to 500 non-empty values per column.
3. It proposes columns whose names and content indicate translatable Chinese text.
4. The user confirms or overrides that selection (human-in-the-loop).
5. Conversation presents primary Character Bible and Target languages actions. Once the required setup is ready, an explicit `Translate` button starts the job.
6. Values are deduplicated, protected, batched, translated, validated, and rejoined to the original rows.
7. The app shows a pre-run workload/token/cost estimate, live batch progress, and safe cancellation between batches.
8. The app reports coverage, API calls, retries, fallback splits, and any failed values. Failed values can be retried without repeating successful work.
9. The user previews and downloads the result, reviews a quality sample, can rerun the full job after submitting a valid glossary change, and can export the execution history. Glossary editing is an explicit state machine: unchanged draft keeps **Apply glossary** disabled; editing enables Apply; successful validation disables Apply and enables **Rerun with current glossary** when a prior result exists; successful rerun disables both again. Before the first result, an applied glossary is used when the user sends target languages in Conversation. Every source column is retained; translated columns use names such as `description__english`.

Projects appear in a ChatGPT-style left sidebar. **New dialogue** is the only entry point that creates or reopens a separate project. Inside an existing project, **Update translation data** replaces that project's source CSV in place while preserving its Character Bible, glossary, target languages, instructions, conversation, and run history. Because an earlier result no longer matches the replacement source, the result and line-level review state are cleared before the next translation. Every project independently retains its current source file, setup assets, chat history, unsent message draft, preview, metrics, review state, translation result, and downloads for the browser session. Switching projects automatically saves the current draft and restores it when the user returns. Translation runs as a background job, so the user can switch projects immediately after submitting a request; an hourglass marks running work and a blue dot marks a completed unread assistant update. Opening the project clears its unread state. The main view keeps the chronological conversation on the left and a sticky, independently scrollable result workspace on the right. Conversation presents the game-mode readiness message and button-based next steps, then opens translation-data update, Character Bible upload, Target languages, or Terminology rules inline; the workspace shows their current readiness and applied-rule counts without duplicating those setup controls.

General CSVs and game string packages use the same Conversation and Workspace shell. Both provide Translation Setup, terminology, target-language selection, cost preview, background progress, result metrics, failure handling, and a primary **Open localization review** action with download inside the review page. General CSV review compares row, source column, language, original value, AI translation, and status. Character Bible, scene context, game QA, confidence signals, and line-level targeted reruns appear only when the uploaded schema supports game localization; a general product CSV no longer falls back to a separate legacy Preview screen.

The interface uses Streamlit's native `st.chat_message`, `st.chat_input`, bottom container, and sidebar with a project-level theme in `.streamlit/config.toml`. Messages are stored per project and rendered oldest-to-newest, so the latest exchange is always the final item in the scrollable history. Game-mode readiness and missing-character guidance are durable assistant messages that are appended only when their status changes. A stable bottom action tray sits outside that history: the initial Translation data, Character Bible, and Target languages actions—and later Terminology rules (including the glossary editor), workload preview, Translate, and completed-result actions—remain directly above the native chat input instead of being reinserted as newer messages on every rerun.

Terminology rules are opened from the guided Conversation step. `Rerun with current glossary` stays hidden until a completed result exists and the user has entered glossary content; it then appears beside `Apply glossary` and becomes available only after the change validates successfully.

The additional example [`samples/additional-examples/sample_products.csv`](samples/additional-examples/sample_products.csv) has 120 data rows for a quick non-game assessment demo.

## Mobile-game localization mode

When a CSV contains a recognizable string identifier plus source column (for example `key` + `source`) or speaker and dialogue columns (for example `speaker` + `source_text`), the app automatically opens the game-localization workflow. A speaker column is optional, so UI and system-string packages do not need a Character Bible. Source text may be Chinese, English, or Japanese; the provider translates from the detected source language into the languages selected in Conversation.

Recommended dialogue schema:

| Column | Purpose |
|---|---|
| `line_id` / `key` / `string_id` | Stable, unique string identifier |
| `source_text` / `source` / `en` / `ja` | Source text to translate |
| `target` / locale column | Existing target value, preserved as input data |
| `scene_id` / `screen` / `location` | Scene, screen, or chapter grouping |
| `speaker` | Optional speaking character |
| `listener` | Addressee |
| `emotion` | Acting/emotional direction |
| `context` / `description` / `note` | Read-only context supplied by the developer/publisher |
| `character_limit` / `char_limit` / `max_length` | Optional target UI limit |
| `status`, `platform`, `plural`, `screenshot_reference` | Preserved workflow metadata |

The included [`samples/additional-examples/string_package_context_demo.csv`](samples/additional-examples/string_package_context_demo.csv) shows a realistic `key + source + target + context + limit` delivery, with [`samples/additional-examples/string_package_character_bible_demo.csv`](samples/additional-examples/string_package_character_bible_demo.csv) providing optional guidance for Mira. [`samples/interview/large_context_localization_demo.csv`](samples/interview/large_context_localization_demo.csv) provides a 120-line Chinese test package: 12 ordered scenes, five characters, speaker/listener relationships, emotions, contextual pronouns, developer notes, placeholders, protected item codes, and character limits. Four deliberately ambiguous `打開它。` rows omit scene, listener, emotion, and developer context so the deterministic Context review queue is non-zero without artificially lowering confidence. Four other rows (`dlg_moon_temple_arrival_02`, `dlg_betrayal_ruins_03`, `dlg_commander_briefing_07`, `dlg_epilogue_temple_06`) carry a deliberately impossible `char_limit` of 5, so the QA issue queue is non-zero as soon as the file is translated, independent of model wording. Import its matching [`samples/interview/large_context_character_bible_demo.csv`](samples/interview/large_context_character_bible_demo.csv); it contains complete guidance for Luna, Player, Commander, Merchant, and Mira. Generic speaker labels such as `System`, `UI`, `Narrator`, and `Tutorial` do not trigger missing-character-guidance warnings. [`samples/additional-examples/game_dialogue_demo.csv`](samples/additional-examples/game_dialogue_demo.csv) and [`samples/additional-examples/character_bible_demo.csv`](samples/additional-examples/character_bible_demo.csv) exercise the smaller character-dialogue example.

Before translation, the setup panel shows the exact scene + speaker work units. After typing target languages in Conversation, **Preview workload & cost** calculates translation values, batches, tokens, and cost without starting the job. Game batches follow those work-unit boundaries while each row still carries its previous/next dialogue context.

### Character Bible and context assembly

The Character Bible is imported from the guided Conversation step instead of requiring a trip to the workspace. Each speaker can define personality, speaking style, relationship, preferred tone, avoided language, catchphrases, and pronouns. For every target line the deterministic orchestrator sends only the relevant speaker profile plus configurable previous/next lines from the same scene, listener, emotion, and scene note.

For string-package-only projects, the review workbench includes a non-blocking deterministic context preflight. It flags missing scene/context, ambiguous short strings, unclear referents, and identical source text used in different scene/speaker/listener combinations. These warnings do not spend model tokens and do not stop the rest of the file from translating.

If one or more speakers have no voice guidance, the first **Send** stops before any model call and shows the missing speakers directly below the button. The button becomes **Continue without Character Bible**; only that explicit second action permits a neutral-voice fallback. Completing the missing profiles restores the normal **Send** action.

Game mode intentionally does not globally deduplicate identical source strings. A cache identity includes source text, speaker, listener, target language, character profile, scene, developer context, neighboring lines, and glossary version. Structured keys are split into transparent hints such as `button · open · chest`; those hints are explicitly treated as unverified and never overwrite developer context. Thus the same word such as “Open” can be localized differently for a chest and an inventory screen.

### Human localization review

The full-screen review workbench uses a page-filling, Excel-like grid. Each row keeps `String ID · Language` in its own column, followed by the original text, editable translation, and reviewer `Notes`. The far-left `Details` checkbox opens exactly one row in a compact full-width panel over the bottom of the grid, so the table never reserves a permanent details column. The two review-action columns are wide enough to show their complete labels.

The grid defaults to `Story context order`, which preserves the source sequence and keeps neighboring dialogue together. Reviewers can instead sort context-risk, QA-flagged, or low-confidence rows to the top without creating separate queues. All rows remain directly editable, so an interviewer does not need to understand an additional review-mode control before using the workbench. The first row in the active ordering is selected automatically so its compact details panel is populated immediately. `Needs context` and `Keep source text` are quick actions rather than a separate status control.

The persistent result workspace turns every output into a translation card rather than presenting a bare model string. Confidence is calculated by the orchestrator from observable evidence—not self-reported by the model—including developer context, addressee or emotion, glossary hits, approved project TM, deterministic ambiguity risks, and QA failures. Cards are labeled High, Medium, or Low and expose the score calculation inputs through provenance.

Each card can show `Developer note`, `ID inference (unverified)`, scene/neighbor grouping, glossary hits, and approved project-TM matches. Ambiguous output remains available as a provisional translation so the file can progress, but the uncertainty is shown explicitly. An approved TM mismatch, glossary violation, placeholder problem, or character-limit failure is surfaced as machine-checkable QA rather than hidden in prose.

The persistent result workspace also includes an editable review table with:

- immutable source and original AI translation;
- a human-readable `QA issue`, deterministic `Suggested fix`, and editable `Final translation`;
- `Needs context` and `Keep source text` quick actions;
- a directly editable translation and reviewer `Notes` column in the main grid;
- a compact Context inspector showing confidence, scene, speaker, listener, emotion, supplied developer note, and deterministic context-risk warnings;
- QA flags and optional AI style score/reason;
- an explicit `Details` selection for focused line-level work.

Saved reviewer edits and intentionally retained source lines are locked during later reruns. The reviewed export keeps the original data and AI columns, then appends reviewed translation, internal workflow state, and reviewer comment columns per target language. Human acceptance and reviewer edit distance are reported as workflow signals.

Developer/publisher context is read-only in the reviewer workflow. A translator can edit the translation, write a `Reviewer note`, mark the line as needing developer context without calling the model, or intentionally keep the source text. Context changes must arrive in an updated source package rather than being silently rewritten by a translator.

All ambiguity flags are grouped and deduplicated into a **Developer question batch** with affected String IDs and source examples. The localization manager can download this as CSV, consolidate the questions with the team, and request an updated source package from the developer. Reviewer-approved translations immediately form a project-local exact-match TM for later cards; context-sensitive duplicate strings remain flagged instead of being blindly overwritten by that TM.

Targeted rerun scopes include selected lines, character, scene, needs-context lines, QA-flagged lines, and lines affected by current glossary terms. The full glossary rerun and failed-value retry remain available separately.

### Game-specific quality checks

Deterministic QA flags empty translations, remaining Chinese in non-Chinese targets, missing placeholders, altered rich-text tags, character-limit violations, required glossary terms, suspicious terminal punctuation, and different source lines collapsed into the same target translation.

The review grid offers context-, QA-, and low-confidence-first sort options without a separate review-mode selector. A visible legend identifies yellow context-review cells, red QA cells, and blue low-confidence cells, while a single merged `Confidence` cell (for example `Low · 42%`) keeps the evidence-based assessment in the main grid without splitting score and level into two columns. A `Length` column shows the current character count against any source-provided limit (for example `28 / 32`) and turns red once the final translation exceeds it, updating live as the reviewer types. `Developer note` surfaces the source package's read-only context directly in the row instead of requiring a trip to Details. Above the QA report, a downloadable Context review expander lists every deterministic context risk with its source, confidence, reason, and available provenance. A safe deterministic suggestion, such as restoring terminal punctuation, includes a **Use fix** action that copies it into `Final translation`; semantic or ambiguous problems display `No safe automatic fix available` and require a reviewer edit or a Needs context decision. The original AI output remains unchanged for auditability. Reviewed CSV exports include separate `__qa_issue`, `__suggested_fix`, and `__final_translation` fields.

Context review, Game QA issues, Failure triage, and the localization review grid use the same ordered information schema: String ID, language, speaker, original, developer note, AI translation, final translation, context review, needs-context decision, QA issue, suggested fix, length, merged confidence, failure reason, and provenance/details. In the interactive grid, the compact **Keep source** action sits inside the Original cell (original text first, action second); read-only reports show the decision as an inline Original-text marker instead of adding a separate column. Fields that do not apply remain blank. The workbench additionally keeps its operational Select, Details, and Notes fields, while Failure triage keeps only final translation and resolution note editable. Scene, speaker/listener, and emotion stay in the compact Details panel rather than the main grid, since Confidence and Developer note are already visible as columns.

Optional AI style evaluation is a separate model pass returning a 0–100 score, reason, and suggestion based on character voice, emotion, relationship, naturalness, and consistency. It is advisory and remains separate from human approval.

## Architecture and agent boundaries

```text
Chat UI / user confirmation
        |
        v
Deterministic orchestration
  - CSV decode and schema inspection
  - Chinese ratio and metadata rules
  - selected-column confirmation
  - deduplication and batching
  - URL/code/placeholder protection
  - glossary and preferred-term substitution
  - workload and cost estimation
        |
        v
Probabilistic translation only
  - LLM receives untrusted strings as data
  - strict indexed JSON response
        |
        v
Deterministic validation
  - exact key/count checks
  - token restoration
  - original-data invariants
  - row/column shape checks
  - coverage and failure report
```

This boundary is intentional. An LLM is valuable for contextual multilingual translation, but it should not decide how to parse the file, align rows, mutate the schema, calculate coverage, or determine whether original data changed.

### Agent-style decomposition

| Stage | Responsibility | Failure behavior |
|---|---|---|
| Inspect | Classify columns using name, type, and Chinese-content ratio | Explain the decision and let the user override it |
| Plan | Parse supported languages and confirm columns | Ask for actionable correction |
| Protect | Replace URLs, emails, codes, placeholders, and glossary terms with stable markers | Deterministic; no model call |
| Translate | Deduplicate and translate indexed JSON batches | Validate, retry with backoff, then recursively split |
| Degrade | Isolate a persistently failing value | Keep the source value and record the error |
| Validate | Verify source equality, row count, and output shape | Stop safely if an invariant fails |
| Deliver | Show preview, metrics, human spot check, CSVs, and execution history | Result remains complete and auditable |

## Detection and preservation rules

Column proposal is explainable rather than hardcoded:

- skip numeric columns;
- skip metadata names such as `id`, `sku`, `url`, `email`, `date`, `price`, and `quantity`;
- sample non-empty text values and propose a column when at least 30% contain Chinese characters;
- expose the proposal in the UI so the user can add or remove columns.

Before translation, URLs, emails, alphanumeric codes such as `SKU-123`, common template placeholders, and user-defined protected terms are replaced with markers and restored afterward. The glossary accepts either `OpenAI` to preserve a term or `會員 | English | member` for a language-specific preferred translation. This makes terminology enforcement deterministic instead of relying only on the model prompt.

Game terminology can optionally include type and notes:

```text
星核 | English | Astral Core | Item | Official name; always capitalize
月神殿 | English | Temple of Luna | Location | Do not abbreviate
```

## Reliability, fallback, and validation

Each model response must be a JSON object with exactly the same indexed keys as the input. Missing keys, extra keys, non-string values, empty translations, provider errors, and malformed JSON are treated as failures.

The failure policy is bounded:

1. retry a transient batch failure once with exponential backoff;
2. stop immediately on non-retryable provider-wide failures such as an invalid API key or model permission error;
3. recursively split a still-failing content/response-validation batch to isolate problematic values;
4. after a single value exhausts retries, copy its source text into the translated column;
5. reduce coverage and add the item to a downloadable failure report.

The user can cancel a running job; cancellation is cooperative, so the current bounded API request is allowed to finish and the workflow stops before the next batch. A cancelled job never replaces the last completed result. The failure report also offers targeted retry, which sends only failed unique values and merges successful retries into the existing translated columns.

The app then verifies that:

- row count is unchanged;
- every original column and value is unchanged;
- the number of appended columns equals `selected columns × target languages`;
- translated unique-value coverage equals successful unique values divided by requested unique values.

Coverage is an operational completeness metric, not a semantic quality score. The UI therefore includes a separate deterministic translation sample that reviewers can mark `Correct` or `Needs revision`, producing a human acceptance rate. Game mode adds an optional, sampled **back-translation check** (`evaluate_hallucination=True`, off by default): a slice of translated lines is round-tripped back into the detected source language and compared to the original with a character-level similarity ratio; anything below the threshold is surfaced as a `possible_hallucination` QA flag through the same review queue as any other QA issue, and lowers that line's confidence score. It is a cheap, deterministic proxy for meaning drift, not a labeled-set metric, and it costs one extra model call per sampled line, so it stays opt-in. Production quality evaluation should additionally use a labeled multilingual test set and metrics appropriate to the content: human adequacy/fluency review, terminology accuracy, COMET or BLEURT, segmented by language, field, content length, and model version. [`evaluation/eval_set.csv`](evaluation/eval_set.csv) and [`evaluate.py`](evaluate.py) are a small step in that direction — a hand-labeled 16-line gold set plus a standalone harness that scores a candidate model/prompt's raw output on exact-match rate, similarity, placeholder preservation, and glossary compliance before it is trusted enough to sit behind this project's deterministic protection layer. Run it with `python evaluate.py`; `tests/test_evaluation.py` asserts the harness actually catches two seeded regressions rather than trivially reporting 0% or 100%.

Every attempt records timestamps, outcome, mode, model, languages, columns, glossary count, estimated tokens/cost, API calls, coverage, duration, and error detail. This execution history can be downloaded as CSV. Cost is deliberately labeled as an estimate: characters are converted to approximate tokens, batch prompt overhead is included, and rates are configurable because provider billing can change.

## Model routing and orchestration

Every batch call can escalate from the default model to a stronger `escalation_model` (`OPENAI_ESCALATION_MODEL`, unset by default) on two triggers: reactively, after a batch's first attempt fails and is being retried, and proactively, before the first attempt, for signals available without calling a model at all. The generic CSV flow uses text length as that signal (a batch containing any value longer than `long_text_threshold` characters); game mode uses the existing deterministic `game_context_risks` flags (missing scene/context, ambiguous short strings, unclear referents) computed before any translation call. This keeps the routing decision cheap and explainable rather than adding a model call just to decide which model to call. `TranslationMetrics.escalated_batches` counts how many batches were escalated, so a run's execution history shows how much of the job used the stronger model.

The same retry/split/degrade decision that `_translate_batch` (`translation_agent.py`) and `_batch`/`_call` (`game_localization.py`) already execute is also expressed as an explicit, named state graph:

```text
translate → validate → success
                     └→ retry  → translate | split
                     └→ split  → translate (recursively, on each half)
                                → retain_source (once a single value still fails)
```

`BATCH_STATE_GRAPH` in `translation_agent.py` documents this shape as data, and `TranslationMetrics.state_events` records the transitions a run actually took, so the graph is provable from a real run rather than aspirational. This is intentionally a plain dict and a bounded event log instead of a LangGraph `StateGraph`: the graph is small, fully deterministic, and has no need for persistence or human-in-the-loop interrupts, so a workflow-framework dependency would not earn its cost here. If this agent grew a second, less deterministic seam — for example, letting the model itself decide whether to request more context instead of only flagging it for a human — that seam is where introducing LangGraph (or a similar orchestrator) would start to pay for itself, and the states/transitions above map directly onto graph nodes and edges if that happens.

**A finding from live testing, not just theory:** running this against a real key surfaced two distinct reliability failure modes on the same "Model response keys did not match the requested game batch" / "Model returned a missing game translation" errors. Large batches occasionally drop or blank one item out of many. The opposite extreme also fails: `game_context_risks` rows with no scene/listener in common with any other line end up alone in a batch of one (see `group_game_records`'s scene+speaker grouping), and a batch of exactly one item is *more* likely to break the strict indexed-JSON contract, not less — the model sometimes replies with the bare translation instead of the `{"0": "..."}` shape a single-item request still requires. Escalating to a stronger model on retry recovers most of these, but the deeper fix is a **minimum batch size**: never send a lone item as its own batch — merge it into a neighboring batch instead, or accept the small loss of context isolation. This is exactly the size of unglamorous, empirical fix that a coarse, upfront batch-size setting won't reveal; it only showed up by watching real state-graph traces across several live runs.

## Why not RAG

Every lookup this product needs is an exact, structured key match: glossary terms, the project's approved translation memory, and the Character Bible are all looked up by an exact source-text/speaker/language key (see `approved_translation_memory` and `glossary_replacements`). None of that requires semantic similarity search over unstructured text, so a vector database or RAG pipeline would add latency, cost, and an unnecessary failure mode without improving translation quality. RAG would earn its place if the product needed to retrieve from large, unstructured, evolving text — for example a publisher's full style guide, a game's lore wiki, or thousands of historical localization decisions — where the relevant passage cannot be found by an exact key and has to be found by meaning. The closest such need in this product is finding a *similar but not identical* prior translation (a near-duplicate string translated slightly differently before); that would be a natural, small-scale place to try embedding similarity over the existing exact-match TM before reaching for a full vector database.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

Optional environment settings:

```bash
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_TIMEOUT_SECONDS="30"
export TRANSLATION_BATCH_SIZE="25"
export TRANSLATION_MAX_RETRIES="1"
export OPENAI_INPUT_COST_PER_MILLION="0.15"
export OPENAI_OUTPUT_COST_PER_MILLION="0.60"
export OPENAI_ESCALATION_MODEL="gpt-4o"            # unset by default: no escalation, no extra cost
export TRANSLATION_HALLUCINATION_CHECK="0"         # game mode only; "1" enables the sampled back-translation check
export TRANSLATION_HALLUCINATION_SAMPLE_RATE="0.2"
export TRANSLATION_HALLUCINATION_THRESHOLD="0.45"
```

### No-cost UI demo mode

Demo Mode exercises the complete interface without an API key or any OpenAI requests. Its output is intentionally synthetic rather than a real translation. A configurable delay makes progress and cancellation visible, while the literal marker `[FAIL]` creates an isolated value failure on a normal run. Choosing **Retry failed values only** automatically disables that artificial failure for the retry job, so the failed value succeeds without changing the glossary.

```bash
export TRANSLATION_DEMO_MODE="1"
export DEMO_TRANSLATION_DELAY_SECONDS="1"
streamlit run app.py
```

Upload [`samples/feature-tests/feature_test.csv`](samples/feature-tests/feature_test.csv) and translate it. The first run creates a downloadable failure; choosing **Retry failed values only** sends only the failed values, and the Demo backend allows that retry to succeed automatically. Disable the mode before real translation:

```bash
unset TRANSLATION_DEMO_MODE
unset DEMO_TRANSLATION_DELAY_SECONDS
```

Alternatively, create `.streamlit/secrets.toml` locally:

```toml
OPENAI_API_KEY = "your-key"
```

That file and `.env` are ignored by Git. Never commit an API key.

## Test

Tests use a fake translation backend, so they are fast, deterministic, and do not consume API credits.

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers Chinese-column detection, conversational multi-language parsing, protected tokens, glossary enforcement, workload/cost estimation, cancellation, a 125-row multi-language batch, source preservation, partial-failure degradation, targeted retry, background execution, file switching, per-file draft recovery, model-routing escalation (on retry and on a deterministic complexity/context-risk signal), the state-graph event trace, the sampled back-translation check, and the standalone evaluation harness (`python evaluate.py` for a printed report).

For a live smoke test, run the app with an API key, upload [`samples/additional-examples/sample_products.csv`](samples/additional-examples/sample_products.csv), keep `product_name` and `category` selected, and enter `Translate to English and Japanese`.

For a no-cost game workflow test, enable Demo Mode, upload [`samples/additional-examples/game_dialogue_demo.csv`](samples/additional-examples/game_dialogue_demo.csv), import [`samples/additional-examples/character_bible_demo.csv`](samples/additional-examples/character_bible_demo.csv), optionally enable AI style evaluation, and enter `Translate to English and Japanese`. Review/edit rows, approve at least one line, select another line, and exercise each targeted rerun scope. Demo output is synthetic but uses the complete context, QA, review, locking, and export pipeline.

For game failure recovery, upload [`samples/feature-tests/game_failure_test.csv`](samples/feature-tests/game_failure_test.csv) in Demo Mode and translate it to English. The line containing `[FAIL]` fails on the first run while the other lines succeed. Use **Retry failed values only**; the artificial failure is disabled for that targeted retry, coverage returns to 100%, and successful lines are not translated again.

The failure report is also an interactive triage table. When failures exist, it appears immediately after the result summary, followed by the expanded **Translation setup**, and then the localization review workbench. Reviewers do not classify a separate action: the app detects a Character Bible edit, an applied glossary change, or a manual translation. The reviewer manually selects a structured **Resolution note** such as `Character voice adjusted`, `Terminology corrected`, or `Translated manually by reviewer`; selecting `Source text kept intentionally` directly performs the Skip behavior, so no separate Skip control is needed. **Apply resolution** stays disabled until both a real resolution and its note are present. Applied decisions remain visible in Failure resolution history.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository and choose `app.py`.
3. Add `OPENAI_API_KEY = "..."` under the app's Secrets settings.
4. Optionally add `OPENAI_MODEL`, `OPENAI_TIMEOUT_SECONDS`, `TRANSLATION_BATCH_SIZE`, `TRANSLATION_MAX_RETRIES`, `OPENAI_ESCALATION_MODEL`, or `TRANSLATION_HALLUCINATION_CHECK` in the deployment environment.
5. Deploy and run the sample-file smoke test.

No key is sent to the browser or stored in the output CSV.

## Trade-offs

- **Streamlit over a workflow platform:** fastest path to a credible upload/chat/preview/download product; the engine remains framework-independent.
- **Rule-based detection over an LLM classifier:** cheaper, reproducible, explainable, and easy to override. It may miss very sparse Chinese content, so the UI shows every column.
- **Unique-value batching:** materially reduces latency and token cost for categorical data. It keeps translations consistent within each source column.
- **One request per batch, sequentially:** simple and friendly to provider rate limits. Concurrency would improve throughput but requires explicit rate and budget controls.
- **Retain source on terminal failure:** prioritizes complete row alignment and recoverability over silently blank output. Coverage and the failure CSV make the degradation visible.
- **Cooperative cancellation:** avoids unsafe thread termination and preserves completed results, but an in-flight provider call cannot be interrupted immediately.
- **Character-based token estimate:** fast and provider-independent, but approximate. Production billing reconciliation should use actual provider usage metadata.
- **Session-level execution history:** good for a demo and single browser session; it is not a durable audit database.
- **Escalate-on-signal model routing over a learned router:** a stronger model kicks in only for retries and rows with an existing deterministic risk signal (text length, or game mode's context-risk flags), so most volume stays on the cheap model. It is coarser than a trained routing model, but it is free, explainable, and reuses signals the app already computes.
- **Sampled back-translation over a full second pass:** checking every line would double translation cost. Sampling a configurable slice keeps the hallucination signal directionally useful without doubling the job's cost, at the price of not catching every drifted line.
- **Hand-rolled state graph over LangGraph for batch retry/split:** the graph is small, fully deterministic, and needs no persistence or human-in-the-loop interrupts, so the dependency would not earn its cost yet; see "Model routing and orchestration" above for where that would change.

## Scaling and production improvements

For larger or business-critical jobs, the next steps would be:

- move work to a durable queue with resumable checkpoints and idempotency keys;
- stream status from background workers and persist per-batch results rather than holding the entire job in session memory;
- add rate-aware concurrency, quotas, and actual usage reconciliation; extend model routing beyond the current retry/context-risk signals to language- and provider-level routing;
- enforce a minimum batch size (merge a lone isolated line into a neighboring batch instead of sending it alone) — see "Model routing and orchestration" above for why single-item batches are also a reliability risk, not just very large ones;
- extend the back-translation check (or replace it) with COMET/BLEURT and a labeled per-language test set, and gate model/prompt releases on `evaluate.py`-style thresholds in CI;
- cache translations by normalized source, target language, glossary version, prompt version, and model version;
- version glossaries, add per-locale terminology ownership, and report term-level compliance;
- use structured outputs where supported and attach request IDs for tracing;
- add automatic language identification at cell level for mixed-language columns;
- store no uploaded content by default, encrypt temporary data, redact logs, and define retention controls;
- add authentication, tenant isolation, audit logs, observability, and alerts for latency, cost, coverage, and quality drift;
- grow `evaluation/eval_set.csv` into a versioned, per-language evaluation set and expand it beyond English;
- persist reviewer corrections as opt-in evaluation data and use edit-distance/style trends to prioritize prompt or model changes.

## Project structure

```text
app.py                    Streamlit conversation and result UI
translation_agent.py      Detection, protection, orchestration, model routing, validation, metrics
game_localization.py      Character context, game QA, review, locking, targeted reruns, back-translation check
evaluate.py               Standalone gold-set evaluation harness
evaluation/               Labeled offline evaluation data
samples/
  interview/              Recommended 120-row interview demo and matching Character Bible
  feature-tests/          Small deterministic files for QA, context, failure, and preservation testing
  additional-examples/    Optional product, string-package, and dialogue examples
tests/                    Automated unit, workflow, UI, and evaluation tests
requirements.txt          Runtime dependencies
requirements-dev.txt      Test-only dependencies
```

## Sample and test files

Only the two files under `samples/interview/` are needed for the primary interview walkthrough. The remaining files provide small, reproducible checks for individual behaviors.

| File | Purpose |
|---|---|
| `samples/interview/large_context_localization_demo.csv` | Recommended 120-row Chinese dialogue demo with linked scenes and deliberate context gaps. |
| `samples/interview/large_context_character_bible_demo.csv` | Complete five-character guidance paired with the interview dialogue file. |
| `samples/feature-tests/review_signals_demo.csv` | Produces Low confidence, Context review, QA issue, and Failed value signals in Demo Mode. |
| `samples/feature-tests/context_qa_review_demo.csv` | Exercises context-only, QA-only, overlapping, and passing review cases. |
| `samples/feature-tests/qa_review_demo.csv` | Exercises safe suggested fixes, manual fixes, empty output, and a passing row. |
| `samples/feature-tests/game_failure_test.csv` | Creates one intentional first-run Demo Mode failure for retry and triage testing. |
| `samples/feature-tests/preservation_test.csv` | Verifies URLs, email addresses, SKUs, order codes, and placeholders remain unchanged. |
| `samples/feature-tests/feature_test.csv` | Small general UI test for glossary rules, protected values, failure, and retry. |
| `samples/additional-examples/sample_products.csv` | Non-game 120-row batching and multi-language assessment example. |
| `samples/additional-examples/game_dialogue_demo.csv` | Small character-dialogue example with scenes, emotion, tags, limits, and placeholders. |
| `samples/additional-examples/character_bible_demo.csv` | Four-character guidance paired with the small dialogue example. |
| `samples/additional-examples/integrated_character_bible_demo.csv` | Expanded five-character English guidance retained for comparison and compatibility testing. |
| `samples/additional-examples/string_package_context_demo.csv` | Realistic string-package example with UI strings, dialogue, plurals, and screenshot references. |
| `samples/additional-examples/string_package_character_bible_demo.csv` | Mira and Player guidance paired with the string-package example. |
| `evaluation/eval_set.csv` | Sixteen labeled English/Japanese examples used by `evaluate.py`, not a primary UI demo file. |
