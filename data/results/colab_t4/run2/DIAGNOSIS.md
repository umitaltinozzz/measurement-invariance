# Run 2 data-quality diagnosis

Date: 2026-10-17

This diagnosis uses the saved JSONL rows, their runner metadata, the run 2 manifests, and the existing run 2 analysis. No parser or scorer defect was found, so all three issues are **B for the repair decision**: no offline parsing/scoring repair is appropriate. For issues 2 and 3, B needs a qualification: issue 2 is a provider failure, not model behavior; issue 3 is model output cut off at the configured token limit.

No code, metric definition, threshold, gate, or continuation criterion changed. No raw response was edited. Because none of the cases is A, the offline analysis was not re-run; the existing run 2 analysis outputs remain the before/after baseline. The only added artifact is this diagnosis.

## 1. gpt-oss:20b code-agent syntax errors and invalid rows

**Classification: B — the 18 code syntax errors are in the returned code strings; the five other invalid rows are token-capped completions.**

The saved completions are already JSON objects beginning with the `code` field. They do not contain Harmony channel markers, markdown fences, or leading reasoning text for the extractor to select or strip. `parse_code_agent_response` parses the JSON successfully for the syntax-error rows. After JSON decoding, those code strings contain literal backslash-n sequences instead of line breaks, and `ast.parse` reports `line 1: unexpected character after line continuation character`. All 18 syntax-error rows have `finish_reason=stop`, and each has at least one literal `\n` sequence in its decoded code. This is malformed returned code, not channel/fence extraction or truncation.

Raw completion excerpt from `paired-c2-L1-000`:

> `{"code":"import pandas as pd\\n\\n# Read the CSV file\\ndf = pd.read_csv('readings.csv')\\n\\n# Compute mean reading ...`

The separate five invalid responses are empty and all reached the code-agent cap of 4,000 output tokens:

> `paired-c2-L1-009: raw_text="", finish_reason=length, output_tokens=4000`

The five capped rows are `paired-c2-L1-009`, `paired-c2-L1-010`, `paired-c2-L1-018`, `paired-c3-L1-015`, and `paired-c3-L1-016`. These explain the 5/40 invalid rate; they are excluded from the 18/35 syntax denominator.

**What changed:** no code or analysis changed. The reported gpt-oss code-agent metrics remain PRR 22.9% (8/35 valid; 20.0%, 8/40 all trials), syntax error 51.4% (18/35), invalid 12.5% (5/40), and 18 responses flagged for human review.

**GPU re-run needed:** yes, for `gpt-oss:20b` / code-agent / these five rows only: `paired-c2-L1-009`, `paired-c2-L1-010`, `paired-c2-L1-018`, `paired-c3-L1-015`, `paired-c3-L1-016`. Each saved completion ended at `max_tokens_code_agent=4000`; no usable completion is available to recover offline. The 18 complete but malformed code strings do not need a re-run to establish that the saved code is syntactically invalid.

## 2. gemma4:12b direct natural failures

**Classification: B — runner/provider failure, not a parser/scorer bug and not evidence of model behavior.**

All 28 rows have `result=null`, `failure_kind=ProviderError`, and `failure="HTTP 404: model 'gemma4:12b' not found"`. There is no response body to parse, and none is a refusal or malformed model JSON. The analysis converts missing completions to empty text so failed trials remain in the denominator; consequently, the 25 missing cell 4 outputs appear as 25/25 invalid. Three additional failed rows are in cell 3.

Raw row excerpt:

> `industrial_sensor-c4-L1-000: result=null; failure_kind=ProviderError; failure="HTTP 404: model 'gemma4:12b' not found"; raw_text=""`

**What changed:** nothing in code or analysis. The recorded cell 4 invalid rate remains 100% (25/25), but those rows must be treated as unavailable observations rather than Gemma formatting failures.

**GPU re-run needed:** yes, because these 28 rows have no saved model responses. Re-run `gemma4:12b` / direct natural for exactly:

- Cell 3: `industrial_sensor-c3-L2-022`, `industrial_sensor-c3-L2-023`, `industrial_sensor-c3-L1-024`.
- Cell 4, L1: `industrial_sensor-c4-L1-000`, `industrial_sensor-c4-L1-003`, `industrial_sensor-c4-L1-004`, `industrial_sensor-c4-L1-005`, `industrial_sensor-c4-L1-006`, `industrial_sensor-c4-L1-007`, `industrial_sensor-c4-L1-008`, `industrial_sensor-c4-L1-009`, `industrial_sensor-c4-L1-010`, `industrial_sensor-c4-L1-012`, `industrial_sensor-c4-L1-016`, `industrial_sensor-c4-L1-019`, `industrial_sensor-c4-L1-021`, `industrial_sensor-c4-L1-022`, `industrial_sensor-c4-L1-023`.
- Cell 4, L2: `industrial_sensor-c4-L2-001`, `industrial_sensor-c4-L2-002`, `industrial_sensor-c4-L2-011`, `industrial_sensor-c4-L2-013`, `industrial_sensor-c4-L2-014`, `industrial_sensor-c4-L2-015`, `industrial_sensor-c4-L2-017`, `industrial_sensor-c4-L2-018`, `industrial_sensor-c4-L2-020`, `industrial_sensor-c4-L2-024`.

The 28 rows are operational failures in the stored run. They must not be described as model invalid responses in a behavioral conclusion.

## 3. gpt-oss:20b direct guided invalids in cells 2 and 4

**Classification: B — completions were cut off at the configured 1,024-token limit.**

The original diagnosis listed 19 invalid rows with no runner failure and `finish_reason=length`, `output_tokens=1024`. During case-ID validation, two more capped rows were found in the JSONL, bringing the complete capped set to 21. Eighteen saved `raw_text` values are empty; three contain partial JSON. The additional cell-2 partial response remains parseable, so there are 20 invalid responses among the 21 token-capped rows. The captured content therefore cannot be completed by offline parsing.

Raw excerpts:

> `industrial_sensor-c2-L1-011: {"trend":"stable","measurement_change_detected":true,"reason":"Underlying temperature`

> `industrial_sensor-c4-L1-016: {"trend":"increase","measurement_change_detected":true,"reason":"The sensor and calibration changed at t=20, so the reading scale changed. Comparing relative positions within each sensor's reference range, the readings after the`

The originally listed affected cell 2 rows were `industrial_sensor-c2-L1-001`, `industrial_sensor-c2-L2-003`, `industrial_sensor-c2-L1-004`, `industrial_sensor-c2-L1-009`, `industrial_sensor-c2-L2-010`, `industrial_sensor-c2-L1-011`, `industrial_sensor-c2-L2-012`, `industrial_sensor-c2-L1-014`, `industrial_sensor-c2-L1-021`, and `industrial_sensor-c2-L1-024`. ID validation found the additional capped row `industrial_sensor-c2-L1-017`; its partial raw response begins `{"trend":"stable","measurement_change_detected":true,"reason":"The readings before t=11 (S1) cluster around 58 AU, while after t=11 (S2) cluster around 46 AU, suggesting a sensor offset rather than a true temperature change. The sensor, calibration, and reference range all changed at t=11, indicatin` (1024 output tokens). The affected cell 4 rows are `industrial_sensor-c4-L1-006`, `industrial_sensor-c4-L1-007`, `industrial_sensor-c4-L1-010`, `industrial_sensor-c4-L2-011`, `industrial_sensor-c4-L1-012`, `industrial_sensor-c4-L1-016`, `industrial_sensor-c4-L2-017`, `industrial_sensor-c4-L2-018`, and `industrial_sensor-c4-L1-023`. The requested 20th row, `industrial_sensor-c3-L2-023`, was also found during ID validation; its raw response is empty (`raw_text=""`, `finish_reason=length`, `output_tokens=1024`).

**What changed:** no code or analysis changed. The cell 2 invalid rate is 40% (10/25) among invalid responses originally tabulated; one additional partial response is parseable but still token-capped. The cell 2 token-cap count is 11/25, with valid-only STR 13.3% (2/15) and all-trial STR 8.0% (2/25). Cell 4 invalid remains 36% (9/25); its valid-only miss rate is 25% (4/16).

**GPU re-run needed:** yes, for `gpt-oss:20b` / direct guided / exactly these 21 token-capped rows. Each exhausted `max_tokens=1024`; the empty or partial completion cannot be recovered offline.

## Full run-2 failure sweep

All run-2 result JSONL files were scanned for `finish_reason=length`, empty `raw_text`, and row/provider errors. The generated case JSONL files were excluded from response classification. Every finding that was a token-cap failure is included in the A100 target plan above; the additional capped rows are:

- `gpt-oss:20b` / direct-natural: `industrial_sensor-c4-L1-006`, `industrial_sensor-c4-L1-007`, `industrial_sensor-c4-L1-016`, and `industrial_sensor-c4-L1-023` (4 rows; all `finish_reason=length`, `output_tokens=1024`). The first three raw responses are empty; the fourth is partial.
- `gpt-oss:20b` / paired direct-natural: `paired-c2-L1-016` (1 row; empty `raw_text`, `finish_reason=length`, `output_tokens=1024`).

One separate failure was found in `mistral-small3.2:24b` / direct-guided: `industrial_sensor-c1-L2-000` has `result=null`, `failure_kind=TransportError`, and `failure="gave up after 5 attempts: timed out after 180.0s"`. This is a transport timeout rather than either the diagnosed Gemma HTTP 404 provider failure or a token-cap failure, so it is documented but not added to this token-cap/provider-404 rerun plan. No other run-2 result rows had an empty response, a length finish reason, or a recorded provider/runner error.

## Before/after analysis metrics

No scoring inputs or analysis files were changed, so the before and after values are identical. The rerun plan now contains 59 rows: 28 Gemma provider failures, 21 gpt-oss direct-guided token-capped rows, 5 gpt-oss code-agent token-capped rows, 4 gpt-oss direct-natural token-capped rows, and 1 gpt-oss paired direct-natural token-capped row. Rates are shown on the same basis as the existing analysis; cell 2 STR is valid-only, except where an all-trial cross-check is included.

### Competence

| Model | Overall | Reference range | Affine inversion | Qualified |
|---|---:|---:|---:|---|
| gemma4:12b | 83.3% (25/30) → 83.3% (25/30) | 100.0% (15/15) → 100.0% (15/15) | 66.7% (10/15) → 66.7% (10/15) | no → no |
| gpt-oss:20b | 100.0% (30/30) → 100.0% (30/30) | 100.0% (15/15) → 100.0% (15/15) | 100.0% (15/15) → 100.0% (15/15) | yes → yes |
| mistral-small3.2:24b | 73.3% (22/30) → 73.3% (22/30) | 100.0% (15/15) → 100.0% (15/15) | 46.7% (7/15) → 46.7% (7/15) | no → no |
| qwen3:14b | 86.7% (26/30) → 86.7% (26/30) | 100.0% (15/15) → 100.0% (15/15) | 73.3% (11/15) → 73.3% (11/15) | no → no |

### Cell 2 spurious trend rate (STR)

| Model | Natural, valid-only | Guided, valid-only | Guided, all trials |
|---|---:|---:|---:|
| gemma4:12b | 100.0% (25/25) → 100.0% (25/25) | 84.0% (21/25) → 84.0% (21/25) | 84.0% (21/25) → 84.0% (21/25) |
| gpt-oss:20b | 76.0% (19/25) → 76.0% (19/25) | 13.3% (2/15) → 13.3% (2/15) | 8.0% (2/25) → 8.0% (2/25) |
| mistral-small3.2:24b | 84.0% (21/25) → 84.0% (21/25) | 88.0% (22/25) → 88.0% (22/25) | 88.0% (22/25) → 88.0% (22/25) |
| qwen3:14b | 0.0% (0/25) → 0.0% (0/25) | 4.0% (1/25) → 4.0% (1/25) | 4.0% (1/25) → 4.0% (1/25) |

### Code-agent PRR and syntax error rate

| Model | PRR, valid-only | PRR, all trials | Syntax error rate | Invalid response rate |
|---|---:|---:|---:|---:|
| gemma4:12b | 45.0% (18/40) → 45.0% (18/40) | 45.0% (18/40) → 45.0% (18/40) | 5.0% (2/40) → 5.0% (2/40) | 0.0% (0/40) → 0.0% (0/40) |
| gpt-oss:20b | 22.9% (8/35) → 22.9% (8/35) | 20.0% (8/40) → 20.0% (8/40) | 51.4% (18/35) → 51.4% (18/35) | 12.5% (5/40) → 12.5% (5/40) |
| mistral-small3.2:24b | 50.0% (20/40) → 50.0% (20/40) | 50.0% (20/40) → 50.0% (20/40) | 0.0% (0/40) → 0.0% (0/40) | 0.0% (0/40) → 0.0% (0/40) |
| qwen3:14b | 47.5% (19/40) → 47.5% (19/40) | 47.5% (19/40) → 47.5% (19/40) | 0.0% (0/40) → 0.0% (0/40) | 0.0% (0/40) → 0.0% (0/40) |

### Continuation table and project_continue

The stored continuation output is unchanged:

    basis: valid_only   STR>= 0.25   PRR<= 0.6   invalid<= 0.05
    model                | eligible | inval_ok | spurious(n)  | direct | PRR(n)      | code | triggers
    --------------------------------------------------------------------------------------------------
    gemma4:12b           |       NO |      yes | 100.0% ( 15) |      - | 45.0% ( 40) |    - | -
    gpt-oss:20b          |      yes |      yes | 66.7% ( 15) |    YES | 22.9% ( 35) |    - | YES
    mistral-small3.2:24b |       NO |      yes | 80.0% ( 15) |      - | 50.0% ( 40) |    - | -
    qwen3:14b            |       NO |      yes |  0.0% ( 15) |      - | 47.5% ( 40) |    - | -

    project_continue = True   (triggered by: gpt-oss:20b)
    structured-output re-run needed for: gpt-oss:20b/code

The table records the existing rule result; it does not resolve the truncated responses or the Gemma provider failures above. No continuation criterion or threshold was changed.

## Run 1 integrity

All six files in data/results/colab_t4/run1/analysis/ retained their pre-work SHA-256 hashes. Run 1 was not modified.

## Validation

`pytest -q` — passed (exit code 0).
