# Colab T4 pilot results

## Setup and provenance

The completed pilot used four Ollama instruction models with Q4_K_M quantization on a Colab Tesla T4 GPU. The manifests record runs on 2026-10-16 UTC, from 12:30:39.015194 through 15:09:26.474908. Direct natural and guided arms used the same pilot dataset; competence and paired code-agent arms used their separately generated datasets.

| Artifact | Dataset SHA-256 | Config SHA-256 |
|---|---|---|
| Pilot dataset (direct natural and guided) | `9dcc74b88d04a0d9473bdc93fe7867113dc9ff94d21878bb77e8f86e9b2470a9` | `5ff7c990155cf9e06212c90c595584b78c4d663db2443246f4ca250317b8d6f6` |
| Competence dataset | `7fdd5dcd2f8ebea11077ef2ede6816c66d4f4846f32ed14dc494445280157584` | `5ff7c990155cf9e06212c90c595584b78c4d663db2443246f4ca250317b8d6f6` |
| Paired direct and code-agent dataset | `7ca1c7468b5690f4841c36d6276b3dd2576a25abf57425dca7fcff05c2b51f52` | `5ff7c990155cf9e06212c90c595584b78c4d663db2443246f4ca250317b8d6f6` |

The four models were `gemma2:9b-instruct-q4_K_M`, `llama3.1:8b-instruct-q4_K_M`, `mistral-nemo:12b-instruct-2407-q4_K_M`, and `qwen2.5:14b-instruct-q4_K_M`. The run manifests report all planned rows complete with no failed rows.

## Competence pre-test

| Model | Overall | Reference range | Affine inversion | Invalid | Qualified |
|---|---:|---:|---:|---:|---|
| Gemma 2 9B | 66.7% (20/30) | 93.3% (14/15) | 40.0% (6/15) | 0.0% (0/30) | No |
| Llama 3.1 8B | 56.7% (17/30) | 80.0% (12/15) | 33.3% (5/15) | 0.0% (0/30) | No |
| Mistral Nemo 12B | 60.0% (18/30) | 100.0% (15/15) | 20.0% (3/15) | 0.0% (0/30) | No |
| Qwen 2.5 14B | 80.0% (24/30) | 100.0% (15/15) | 60.0% (9/15) | 0.0% (0/30) | No |

## Direct arms by cell

Rates are valid-only, as in the generated analysis. Spurious Trend Rate (STR) and Miss Rate (MR) are reported together; `n/a` means that metric has no applicable denominator for that cell. Each cell has 25 trials per model and prompt variant. Invalid response rate was 0.0% (0/25) for every row below.

| Model | Prompt | Cell 1 STR / MR | Cell 2 STR / MR | Cell 3 STR / MR | Cell 4 STR / MR |
|---|---|---:|---:|---:|---:|
| Gemma 2 9B | Natural | 52.0% (13/25) / n/a (0/0) | 80.0% (20/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Gemma 2 9B | Guided | 76.0% (19/25) / n/a (0/0) | 88.0% (22/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Llama 3.1 8B | Natural | 68.0% (17/25) / n/a (0/0) | 100.0% (25/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Llama 3.1 8B | Guided | 36.0% (9/25) / n/a (0/0) | 92.0% (23/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Mistral Nemo 12B | Natural | 36.0% (9/25) / n/a (0/0) | 60.0% (15/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Mistral Nemo 12B | Guided | 36.0% (9/25) / n/a (0/0) | 36.0% (9/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Qwen 2.5 14B | Natural | 0.0% (0/25) / n/a (0/0) | 24.0% (6/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |
| Qwen 2.5 14B | Guided | 0.0% (0/25) / n/a (0/0) | 12.0% (3/25) / n/a (0/0) | n/a (0/0) / 0.0% (0/25) | n/a (0/0) / 0.0% (0/25) |

## Code-agent arm

The arm **ran static-only**: `execution_enabled` was false and generated code was not executed. Invalid response rate was 0.0% (0/40) for every model. Gemma 2 had 100.0% (40/40) syntax errors and Llama 3.1 had 87.5% (35/40); their code-arm metric values are uninterpretable because most or all generated code did not parse.

| Model | Provenance retention | Effective provenance use | STR | MR | Syntax errors |
|---|---:|---:|---:|---:|---:|
| Gemma 2 9B | 0.0% (0/40) | 0.0% (0/40) | 50.0% (10/20) | 0.0% (0/20) | 100.0% (40/40) |
| Llama 3.1 8B | 2.5% (1/40) | 2.5% (1/40) | 100.0% (20/20) | 0.0% (0/20) | 87.5% (35/40) |
| Mistral Nemo 12B | 2.5% (1/40) | 2.5% (1/40) | 100.0% (20/20) | 0.0% (0/20) | 0.0% (0/40) |
| Qwen 2.5 14B | 50.0% (20/40) | 50.0% (20/40) | 75.0% (15/20) | 0.0% (0/20) | 0.0% (0/40) |

## Continuation verdict

`project_continue = False`. No model passed the competence gate, so no model was eligible for either continuation trigger. The analysis reports all models as ineligible; invalid-rate checks passed for each, but did not override the competence failure.

| Model | Competence eligible | Invalid-rate gate | Direct trigger | Code trigger |
|---|---|---|---|---|
| Gemma 2 9B | No | Yes | ? | ? |
| Llama 3.1 8B | No | Yes | ? | ? |
| Mistral Nemo 12B | No | Yes | ? | ? |
| Qwen 2.5 14B | No | Yes | ? | ? |

## What this pilot does and does not show

- Under the pre-registered rule, cell-2 results from models that failed competence cannot be interpreted as measurement-invariance failures.
- Cell 1 is the noise control. Gemma 2, Llama 3.1, and Mistral Nemo have elevated cell-1 spurious rates, so their cell-2 rates are confounded with reading noise as signal.
- The only clean contrast in this pilot is Qwen 2.5 14B: cell 1 is 0.0% for both prompt variants, while cell 2 is 24.0% natural and 12.0% guided. Qwen also failed the competence gate, so this contrast is descriptive and does not qualify as a continuation result.
- The next step is to evaluate larger or frontier models that pass competence, alongside an execution-enabled code-agent arm.

These results describe this pilot's four quantized Ollama models on the generated benchmark. They do not establish a general claim about LLMs or demonstrate that the broader research question has been answered.

## Artifacts

The complete runner JSONL files, manifests, generated analysis, and chart are in [`data/results/colab_t4/run1/`](data/results/colab_t4/run1/). The analysis was generated from the runner outputs and adjacent manifests.
