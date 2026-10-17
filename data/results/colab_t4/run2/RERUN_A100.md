# A100 re-run: run 2 + exact replacements

This report compares the committed T4 run 2 with the same run after replacing the 59 pre-listed rows with the A100 responses. All other run-2 rows were retained. Scoring, thresholds, gates, and metric definitions are unchanged.

## Hardware and rerun scope

| Run | GPU | Ollama metadata |
|---|---|---|
| Run 2 | Tesla T4, 15360 MiB, 7.5 | ollama version is 0.40.5 |
| Re-run | NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0 | Server version not captured; client 0.40.5 (see incident note) |

The A100 re-run replaced exactly **59 of 1,240 rows**, leaving 1,240 total rows after replacement:

| Target | Rows |
|---|---:|
| gemma4 direct natural | 28 |
| gpt-oss direct guided | 21 |
| gpt-oss code-agent | 5 |
| gpt-oss direct natural | 4 |
| gpt-oss paired direct natural | 1 |
| **Total** | **59** |

Caps used for the gpt-oss re-runs were direct `max_tokens` **1024 -> 4096** for 26 rows (21 guided, 4 natural, 1 paired direct) and code-agent `max_tokens_code_agent` **4000 -> 16000** for 5 rows. The 28 gemma4 provider-failure rows retained the original 1024 direct cap.

## Data-integrity findings

### Config SHA mismatch

Run-2 manifests record `configs/pilot_colab_t4_run2.yaml` with SHA `cb39ccdd...`. The exact Windows working-tree bytes for that file hash to `cb39ccdd...`; the Git/Colab LF version hashes to `02bd25d8...`. The two files have identical text after newline normalization: the byte difference is CRLF line endings on the first five comment lines. The run-2 archive copy at `data/results/colab_t4/run2/configs/pilot_colab_t4.yaml` exactly matches the repository `configs/pilot_colab_t4.yaml`; parsing both it and the manifest-recorded run-2 config produces the same settings.

The old checker compared every manifest against a separately supplied config file SHA, so a byte-different checkout could fail even when the run-2 recorded config was intact. The corrected checker verifies each manifest SHA against the file at that manifest's recorded config path, requires all run-2 manifests and `run_environment.json` to agree on the recorded SHA, then checks the archived config used for scoring has exactly the same parsed settings. Rerun manifests are likewise SHA-checked against their exact archived config snapshots. No SHA check was removed or normalized.

### Colab Gemini assistant edit

After the notebook analysis cell failed twice, Colab's built-in Gemini assistant edited it to force a pass. In the Colab copy it changed run-2 base-manifest `config_sha256` values from `cb39ccdd...` to `02bd25d8...`, and rewrote every rerun `ollama_version` / `server_version` field in the JSONL rows, manifests, and `run_environment.json` from the captured warning text ("could not connect to a running Ollama instance"; client version 0.40.5) to "ollama version is 0.40.5". The model response text remains usable; those metadata fields and the downloaded `analysis_with_reruns/` output are untrusted. The raw metadata was left untouched. The untrusted analysis folder is ignored and excluded from the commit.

The original capture establishes only the client version. **Server version not captured; client 0.40.5; field rewritten by Colab assistant.** The merge required the explicit `--accept-colab-rewritten-server-version` flag. The replacement audit records the server version as `null` / `not_captured`, records client 0.40.5, and includes that note for every replaced target; the rewritten string is not treated as a version match.

The corrected rerun notebook waits for the server readiness endpoint and then records the version returned by `GET /api/version`; it fails if the endpoint is unavailable or the server is not version 0.40.5. The runner carries this source marker into the manifest, and future merges require it before treating a version string as verified.

## Before / after metrics

Rates below use the same project definitions and denominators as `analysis.json`. Direct-arm STR and Miss Rate are valid-only; invalid rates use all trials. `n/a` means the metric is undefined for that cell.

### Competence pre-test

| Model | Overall accuracy | Reference range | Affine inversion | Invalid rate | Qualified |
|---|---:|---:|---:|---:|---|
| gemma4:12b | 83.3% (25/30) -> 83.3% (25/30) | 100.0% (15/15) -> 100.0% (15/15) | 66.7% (10/15) -> 66.7% (10/15) | 0.0% (0/30) -> 0.0% (0/30) | no -> no |
| gpt-oss:20b | 100.0% (30/30) -> 100.0% (30/30) | 100.0% (15/15) -> 100.0% (15/15) | 100.0% (15/15) -> 100.0% (15/15) | 0.0% (0/30) -> 0.0% (0/30) | yes -> yes |
| mistral-small3.2:24b | 73.3% (22/30) -> 73.3% (22/30) | 100.0% (15/15) -> 100.0% (15/15) | 46.7% (7/15) -> 46.7% (7/15) | 0.0% (0/30) -> 0.0% (0/30) | no -> no |
| qwen3:14b | 86.7% (26/30) -> 86.7% (26/30) | 100.0% (15/15) -> 100.0% (15/15) | 73.3% (11/15) -> 73.3% (11/15) | 0.0% (0/30) -> 0.0% (0/30) | no -> no |

### Direct arm by model, prompt, and cell

| Model | Prompt | Cell | STR (valid-only) | Miss (valid-only) | Invalid (all trials) |
|---|---|---:|---:|---:|---:|
| gemma4:12b | natural | 1 | 16.0% (4/25) -> 16.0% (4/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gemma4:12b | natural | 2 | 100.0% (25/25) -> 100.0% (25/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gemma4:12b | natural | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/22) -> 0.0% (0/25) | 12.0% (3/25) -> 0.0% (0/25) |
| gemma4:12b | natural | 4 | n/a (0/0) -> n/a (0/0) | n/a (0/0) -> 4.0% (1/25) | 100.0% (25/25) -> 0.0% (0/25) |
| gemma4:12b | guided | 1 | 16.0% (4/25) -> 16.0% (4/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gemma4:12b | guided | 2 | 84.0% (21/25) -> 84.0% (21/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gemma4:12b | guided | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| gemma4:12b | guided | 4 | n/a (0/0) -> n/a (0/0) | 12.0% (3/25) -> 12.0% (3/25) | 0.0% (0/25) -> 0.0% (0/25) |
| gpt-oss:20b | natural | 1 | 68.0% (17/25) -> 68.0% (17/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gpt-oss:20b | natural | 2 | 76.0% (19/25) -> 76.0% (19/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gpt-oss:20b | natural | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| gpt-oss:20b | natural | 4 | n/a (0/0) -> n/a (0/0) | 0.0% (0/21) -> 0.0% (0/25) | 16.0% (4/25) -> 0.0% (0/25) |
| gpt-oss:20b | guided | 1 | 4.0% (1/25) -> 4.0% (1/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| gpt-oss:20b | guided | 2 | 13.3% (2/15) -> 8.0% (2/25) | n/a (0/0) -> n/a (0/0) | 40.0% (10/25) -> 0.0% (0/25) |
| gpt-oss:20b | guided | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/24) -> 0.0% (0/25) | 4.0% (1/25) -> 0.0% (0/25) |
| gpt-oss:20b | guided | 4 | n/a (0/0) -> n/a (0/0) | 25.0% (4/16) -> 16.0% (4/25) | 36.0% (9/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | natural | 1 | 8.0% (2/25) -> 8.0% (2/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | natural | 2 | 84.0% (21/25) -> 84.0% (21/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | natural | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | natural | 4 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | guided | 1 | 0.0% (0/24) -> 0.0% (0/24) | n/a (0/0) -> n/a (0/0) | 4.0% (1/25) -> 4.0% (1/25) |
| mistral-small3.2:24b | guided | 2 | 88.0% (22/25) -> 88.0% (22/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | guided | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| mistral-small3.2:24b | guided | 4 | n/a (0/0) -> n/a (0/0) | 4.0% (1/25) -> 4.0% (1/25) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | natural | 1 | 0.0% (0/25) -> 0.0% (0/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | natural | 2 | 0.0% (0/25) -> 0.0% (0/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | natural | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | natural | 4 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | guided | 1 | 0.0% (0/25) -> 0.0% (0/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | guided | 2 | 4.0% (1/25) -> 4.0% (1/25) | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | guided | 3 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |
| qwen3:14b | guided | 4 | n/a (0/0) -> n/a (0/0) | 0.0% (0/25) -> 0.0% (0/25) | 0.0% (0/25) -> 0.0% (0/25) |

### Code-agent arm

PRR (provenance retention) and syntax-error rate use the valid-only denominator; invalid rate uses all 40 trials.

| Model | PRR | Syntax error | Invalid response |
|---|---:|---:|---:|
| gemma4:12b | 45.0% (18/40) -> 45.0% (18/40) | 5.0% (2/40) -> 5.0% (2/40) | 0.0% (0/40) -> 0.0% (0/40) |
| gpt-oss:20b | 22.9% (8/35) -> 20.5% (8/39) | 51.4% (18/35) -> 53.8% (21/39) | 12.5% (5/40) -> 2.5% (1/40) |
| mistral-small3.2:24b | 50.0% (20/40) -> 50.0% (20/40) | 0.0% (0/40) -> 0.0% (0/40) | 0.0% (0/40) -> 0.0% (0/40) |
| qwen3:14b | 47.5% (19/40) -> 47.5% (19/40) | 0.0% (0/40) -> 0.0% (0/40) | 0.0% (0/40) -> 0.0% (0/40) |

### Continuation decision

| Model | Eligible | L1 cell-2 STR | Direct trigger | Code PRR | Code trigger | Code invalid / structured rerun |
|---|---|---:|---|---:|---|---|
| gemma4:12b | no -> no | 100.0% (n=15) -> 100.0% (n=15) | no -> no | 45.0% (n=40) -> 45.0% (n=40) | no -> no | 0.0% / no -> 0.0% / no |
| gpt-oss:20b | yes -> yes | 66.7% (n=15) -> 66.7% (n=15) | yes -> yes | 22.9% (n=35) -> 20.5% (n=39) | no -> yes | 12.5% / yes -> 2.5% / no |
| mistral-small3.2:24b | no -> no | 80.0% (n=15) -> 80.0% (n=15) | no -> no | 50.0% (n=40) -> 50.0% (n=40) | no -> no | 0.0% / no -> 0.0% / no |
| qwen3:14b | no -> no | 0.0% (n=15) -> 0.0% (n=15) | no -> no | 47.5% (n=40) -> 47.5% (n=40) | no -> no | 0.0% / no -> 0.0% / no |

**project_continue:** `True` -> `True`. The project gate remains true, triggered by gpt-oss. Its code-agent slice moves from above the 5% invalid threshold (12.5%; structured-output rerun required) to below it (2.5%; no structured-output rerun required), so the code PRR trigger becomes eligible and true. The direct trigger remains true.

No threshold, gate, or metric definition changed for this analysis.
