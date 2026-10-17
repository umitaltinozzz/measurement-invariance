# Run 3 preregistration: controlled measurement-function invariance

Prepared 2026-10-18. **Ready for external registration; not yet deposited on OSF.**
No run-3 model responses have been collected as part of this preparation. The
authors have seen runs 1 and 2 and the [Barış Şavk review](reviews/2026-10-17_review_baris_savk.md).
Those observations informed this new protocol. Run 1 remains a stop under its
original rule; run 2 remains exploratory. Neither is rescored as confirmatory
evidence under this protocol.

The registration package is this document, the Git commit containing it,
[open-model config](../configs/run3_open.yaml),
[frontier-candidate config](../configs/run3_frontier.yaml),
[dataset hashes](run3_dataset_manifest.json), source code, classifier fixtures,
and [dependency pins](../requirements-lock.txt). Deposit the package with an
external timestamp **before any benchmark response is collected**. Record the
OSF URL and registration identifier in a subsequent administrative update. This
document does not claim that a Git commit alone is an external preregistration.

## 1. Questions, hypotheses and scope

The unit is one synthetic industrial-sensor case answered by one model in one
arm, prompt variant, and repeat. Ground truth is derived from latent states
before observations are generated. Each request is a fresh conversation.

* **H1, primary confirmatory:** under `natural_sustained`, at L1, a model's
  probability of a spurious trend is greater in cell 2 (measurement changes,
  latent state fixed) than in cell 1 (both fixed). Test all seven models; do not
  select them using competence or benchmark outcomes.
* **H2, secondary confirmatory:** on the separate paired L1 cell-2 dataset, the
  executed-code arm has a higher spurious-trend probability than direct
  `natural_sustained` on the same cases. Report improvement as well as worsening;
  a direction opposite H2 is not evidence for H2. Invalid and abstaining answers
  remain visible in the paired transition table.
* **Registered operational code criterion:** correction PRR in cell 2 is at most
  0.60, subject to the quality and abstention gates below. This is a screening
  criterion, not proof of a semantic program defect or a significance test.
* **Exploratory:** old natural versus sustained wording; guided versus sustained;
  L2; competence interactions; cell-4 raw-direction subgroups; declared/executed
  disagreement; reasoning traces; mixed-effects analyses; human recoding.

This is a focused run-3 follow-up, not the full 48-condition clinical/finance/
industrial, L0–L3 benchmark proposed in `proje.md`. It covers industrial L1/L2.
Cell-4 diagnostic subgroups are reported separately but are not forced to equal
size in this follow-up; balancing them is reserved for the full benchmark.

## 2. Models and inference conditions

The intended roster is all four specified open models plus all three frontier
candidates below. The word candidate identifies the proposed external models
before OSF deposit; the deposited roster must be this complete list or an
explicitly amended list registered before collection. There is no outcome-based
replacement or fallback. A retired/unavailable endpoint is reported unavailable.

| Model ID | Provider / pinned upstream | Reasoning setting |
|---|---|---|
| `qwen3:14b` | Ollama, local | `think: true` |
| `gemma4:12b` | Ollama, local | `think: false` |
| `mistral-small3.2:24b` | Ollama, local | `think: false` |
| `gpt-oss:20b` | Ollama, local | `think: medium` |
| `openai/gpt-5.4` | OpenRouter / `openai` | `reasoning.effort: medium`, `exclude: false` |
| `anthropic/claude-sonnet-4.6` | OpenRouter / `anthropic` | same explicit gateway effort |
| `google/gemini-3.1-pro-preview` | OpenRouter / `google-ai-studio` | same explicit gateway effort |

These are fixed model-version candidates, not a claim that each is the latest
release at collection time. OpenRouter uses `allow_fallbacks: false` and
`require_parameters: true`. Its served model/upstream IDs are retained per row.
Unsupported requests are provider failures; do not silently strip reasoning or
schema settings to obtain a response.

All models have initial output budgets of **8,192 tokens for direct/competence**
and **16,384 for code**, including any reasoning tokens. Ollama receives
`num_ctx: 65536`. The frontier config reserves that context envelope for planning;
the gateway does not offer a matching `num_ctx` control. Actual prompts fit well
within the remaining envelope. Local temperature is 0 and the requested seed is
2026101703 + repeat index. Frontier requests deliberately omit temperature and
seed (explicit `null` / `send_seed: false`) and use native sampler defaults; this
avoids pretending that different reasoning APIs share sampling controls. The
request timeout is 3,600 seconds per model, covering the preregistered doubled
code cap at the assumed local throughput.

Reasoning parity means settings are explicit and held constant within each
model across arms/repeats, with sufficient output budgets. It does **not** mean
equal internal computation across architectures. Qwen thinking-on versus the
run-2 thinking-off condition is a deliberate new condition. Ollama's
`message.thinking` is stored as `result.thinking`; gateway `message.reasoning`
and `reasoning_details` are retained when present. Missing traces are null, not
evidence of no reasoning. Traces are never part of the answer parser or primary
grading. Controls follow [Ollama's thinking documentation](https://docs.ollama.com/capabilities/thinking)
and [OpenRouter's reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Before collection, record local model digests, quantization, Ollama version and
hardware in the manifest inputs (`OLLAMA_MODEL_DIGEST`, `OLLAMA_SERVER_VERSION`,
`OLLAMA_SERVER_VERSION_SOURCE`, and `--hardware-json`). Use one fixed local
hardware/software setup and frozen downloaded weights throughout. Record the
Docker image ID; each execution also stores it. A change in weights, upstream,
reasoning, hardware or code revision is a protocol deviation, not a cap rerun.
Hosted aliases may drift without disclosing a weight digest; report this limit.

## 3. Cases, repetitions and stopping

Per model, **per repeat**, collect the following complete design:

| Dataset / arm | Cases and variants | Calls |
|---|---|---:|
| Main direct | 4 cells × (50 L1 + 50 L2) × 3 prompts | 1,200 |
| Paired direct | cells 2 and 3 × 50 L1 × sustained prompt | 100 |
| Paired code | the same 100 cases, executed answer | 100 |
| Competence | 30 reference-range + 30 affine-inversion items | 60 |
| Total | | **1,460** |

The three direct prompts are unchanged `natural`, unchanged `guided`, and new
`natural_sustained`. Each main cell-by-salience slice has 50 cases. The paired
set uses an independent seed namespace and is shared byte-for-byte across arms.
All seven models receive the same datasets.

There are **three independent dataset repeats**, indexed 0, 1, 2, using master
seeds 2026101703, 2026111703, 2026121703. Each case is sampled once in each arm;
repeats use fresh latent/noise draws, not three calls to the same case. Dataset
hashes are frozen in `docs/run3_dataset_manifest.json`. There are 4,380 calls per
model, 17,520 open-model calls, 13,140 frontier calls, **30,660 total**, before
transport retries and cap reruns. Collect all three repeats regardless of any
interim result; there is no early stopping for significance or success.

Repeat 0 supplies the primary decision and confirmatory tests. Repeats 1 and 2
are preplanned replications reported separately; they do not rescue a failed
primary decision. Report whether each model/arm's repeat-0 trigger is reproduced
in both replications. Do not pool repeated observations in a binomial interval.
No sample-size extension, model substitution, or prompt retuning follows results.

Generation retains the existing 30 timesteps, 5% multiplicative Gaussian noise,
mid-window switch, latent levels 40–60 and changes of 15–40%, balanced measurement
direction/gain/offset grid (gains 1.2 and 2; offsets -2, 0, 2). The generator and
analytic ground-truth validators are unchanged for historical configs.

## 4. Rule versions, denominators and competence

The historical continuation implementation is preserved in `src/rule_v1.py`.
Omitting `continuation_criteria.rule_version` selects `rule_v1`. Historical config
bytes, thresholds and result files are unchanged. Run 3 explicitly selects
`rule_v2`; its unused legacy 0.25 STR threshold is retained only for schema
compatibility and never participates in the new decision.

For each model, repeat, **same prompt and same provenance salience**, let
`p2 = spurious2 / valid2` and `p1 = spurious1 / valid1`. The registered primary
slice is `natural_sustained`, L1. `uncertain` is a valid response and remains in
these denominators. Compute `delta = p2 - p1` and a two-sided 95% Newcombe
method-10 interval from the independent Wilson score intervals (no continuity
correction), using z = 1.959963984540054. For Wilson limits `(Lj, Uj)`:

```
lower = delta - sqrt((p2 - L2)^2 + (U1 - p1)^2)
upper = delta + sqrt((U2 - p2)^2 + (p1 - L1)^2)
```

Clip limits to [-1, 1]. An empty denominator produces no interval and no trigger.
This is the Wilson combination, not subtraction of interval endpoints.
Method: [Newcombe (1998)](https://pubmed.ncbi.nlm.nih.gov/9595617/).

**Direct trigger:** `lower > 0` strictly, with all direct L1 cells 1, 2 and 3
eligible. Each needs at least 50 collected cases, invalid rate at most 0.05, and
abstention rate at most 0.50. Exactly 0.50 is allowed; greater than 0.50 is flagged
**non-informative** and cannot trigger. Abstention means `uncertain / valid`, not
`uncertain / non-uncertain` (which would be a contradictory denominator).
Applying the cap to each cell prevents pooled averages hiding blanket skepticism
in cells requiring a decision. Cell 2 also has its own cap even though abstention
can be warranted there. All-trial STR is reported alongside valid-only STR.

**Code trigger:** correction PRR at most 0.60, computed **only on cell 2, L1**,
using classifier v2 and valid primary code responses. Cell 3 never enters its
PRR numerator or denominator. Code cells 2 and 3 must both meet the same size,
invalid and abstention gates; direct control cells 1 and 3 must also be eligible.
Report all-trial PRR and Wilson intervals. Syntax/output/runtime failures cannot
improve the primary score through abstention: they count against the invalid
gate. Exact threshold equality is accepted. Missing controls or incompatible
classifier versions prevent a trigger.

Within one repeat/cohort the software computes the OR of eligible model/arm
triggers. The primary study planning decision is the OR of the two complete
repeat-0 cohort decisions. Missing/capped/unavailable slices yield **incomplete**
and no cohort continuation verdict, not evidence of successful invariance.
The former frontier-only continuation requirement is **explicitly removed for
run 3**: the listed open and frontier models have equal screening status. No
claim about frontier models can be made from the open cohort alone. The OR rule
is a planning signal; it is not a familywise significance test.

**Competence choice:** keep and report the pre-test, but remove it as an
exclusion gate for rule_v2. Recognizing non-comparability in cell 2 does not
require affine inversion, so excluding a model for inversion failure would hide
the measurement-specific effect the review identified. Historical qualification
(overall ≥0.90, each skill ≥0.85, invalid ≤0.05) is still computed, on all trials,
and supplied as a diagnostic. Repeat H1 descriptively within qualified models.
Cell-4 recovery and claims about deployment of an already-demonstrated inversion
skill must be stratified by inversion competence; a failing model's errors do
not establish a capability/deployment dissociation.

## 5. Prompt and code-arm protocol

`natural_sustained` defines a change as a sustained level shift beyond ordinary
fluctuations, comparing typical levels across several consecutive readings on
either side of a possible transition, rather than endpoints. It adds no explicit
normalization instructions. The system role, log introduction and task definition
are shared verbatim with the code prompt. The direct table and code CSV have the
same rounded readings/provenance. Instructions for writing/executing code are
the intended modality difference. Old prompts remain available for comparison.

Code responses contain exactly `code_lines` (an array of source lines), `answer`
(declared conclusion) and `reason`. This avoids asking for literal escaped
newlines inside a single source string. Code uses the Python standard library,
reads `readings.csv`, and prints exactly one JSON object such as
`{"answer": "stable"}`. Accepted values are increase/decrease/stable/uncertain.
No code repair, escape repair, dependency installation, extra print-line
selection, or model feedback loop is applied.

Run-3 configs select `answer_source: executed`. The grader retains both declared
and executed grades and their agreement; a failed execution never falls back
to the declared answer. Old configs remain static/declared. Execution can be
disabled with declared grading in a separately registered exploratory config.

The runner invokes Docker by subprocess: immutable image ID, no network,
non-root UID 65534, all capabilities dropped, no new privileges, read-only root
and input mount, temporary per-case directory, 16 MiB scratch tmpfs, 512 MiB RAM
with no additional swap, one CPU, 10 CPU seconds, 32 processes, 64 descriptors,
1 MiB file limit, 20-second wall timeout and at most 20,000 captured bytes per
stream. On timeout/output overflow it removes the container, not just the client.
The image is built from the digest in `sandbox/Dockerfile`; no image is pulled
implicitly during execution. Host `SubprocessSandbox` is not the run-3 backend.
The base image has no pandas dependency; this matches the new prompt.

Classifier v2 counts recognized range normalization `(reading-low)/(high-low)`
or affine inversion `(reading-offset)/gain` flowing to the printed `answer`.
Selection, metadata printing/dropping, grouping raw means, unrelated conditions,
and discarded corrected values do not count. Provenance-aware abstention is
recorded separately and is not correction. This operational PRR supersedes
v1's mere-column-retention PRR only for run 3.

Validation uses the [18 labelled snippets](../tests/fixtures/correction_labels.json)
and a complete executable normalization fixture. Labels and rationales are
manually specified synthetic gold labels; they are not independent human ratings
of unseen outputs. All 18 must match before collection. The AST screen remains
heuristic: it may miss helpers/dynamic code and does not prove semantic correctness.
Keep `needs_human_review` on every annotation. After collection, hand-code a fixed
sample of the first 10 cell-2 programs by case ID per model/repeat (210 total),
blinded to model name and correctness when possible; report the confusion matrix
and disagreements as an exploratory sensitivity analysis. Do not change the
frozen automatic classifier or silently replace its labels.

## 6. Failure, retry and rerun policy

* Transport failures (timeouts, 429, retryable server failures) receive up to four
  retries within the same logical trial, with waits of 1, 2, 4, 8 seconds. Provider
  errors without a completion are stored as unavailable, not model-invalid.
* On resume, unavailable provider rows are re-asked with identical settings and
  archived to `.provider_failures.jsonl`. At most two such resume re-asks per
  logical trial are permitted. Remaining failures stay unavailable and make the
  design incomplete. The audit plus active row preserve attempt history.
* Malformed content, refusal, invalid answer JSON, code syntax errors and program
  failures are observations; do not re-ask them. A sandbox infrastructure failure
  retains the original completion and makes the analysis incomplete; it must not
  buy another model response. Any recovery execution of that same source must be
  archived as an explicit execution-only deviation.
* If **any** row reports `length`/`max_tokens`, archive the entire initial slice
  and run **every case in that slice** once at stage 1. A slice is model × repeat
  × dataset × arm × prompt variant (thus a main direct slice contains all four
  cells and both salience levels). Never splice individual capped rows into the
  initial slice. The initial slice may be partial if interrupted; the replacement
  must contain the whole planned slice.
* The sole cap escalation is fixed now: 16,384 direct/competence, 32,768 code.
  Keep reasoning, data, sampler, hardware and upstream fixed. Stage-1 replaces
  stage-0 as a whole and both remain archived. A still-capped stage-1 slice is
  excluded from decision eligibility and the cohort marked incomplete. No stage 2.
* Structured output is enabled from the start; invalid-rate excess alone does
  not authorize a content rerun. Report the failed quality gate.

`analyze_run3` verifies dataset/config hashes, registered dataset contents for
the repeat, actual request settings and prompt fingerprints, full design
coverage, stage replacement conditions, and unique trial IDs. It refuses pooled
repeats. Re-run analysis on one cohort/repeat directory, never on all repeats.

## 7. Statistical analysis and reporting

1. Report each cell × model × prompt × salience × repeat, with numerators,
   denominators, valid-only and all-trial rates and Wilson 95% intervals: STR,
   miss, wrong direction, over-abstention, total uncertain, invalid, blanket
   skepticism, measurement-change detection where asked, and L1 cell-4 recovery.
   Provider-unavailable counts are separate. Code tables show declared and
   executed outcomes, disagreement, syntax/execution failures and correction PRR.
2. Report H1's Newcombe intervals and rule_v2 decisions independently per model.
   For formal confirmatory H1 claims use a one-sided Fisher exact test of cell 2
   greater than cell 1, valid-only, repeat 0, sustained L1; apply Holm correction
   over all seven preregistered models at familywise alpha 0.05. Ineligible or
   unavailable comparisons receive p=1 for family accounting and no positive
   claim. The 95% screening intervals remain unadjusted and clearly labelled.
3. H2 uses genuine paired case IDs only. Report exact two-sided McNemar tests of
   spurious-trend indicators (primary paired endpoint), and of correctness
   (secondary descriptive endpoint); show the entire outcome transition table,
   uncertain/invalid shares, and unpaired counts. Holm-adjust the seven repeat-0
   spurious tests as a separate secondary family. Require a worsening sign for
   H2 support. Abstention-to-trend transitions are not described as loss of
   previously demonstrated correction. Do not compare main-dataset direct rates
   with paired code rates as though they were paired.
4. Show repeats 1 and 2 separately, with all registered repetitions included.
   Formal family adjustments and the cross-cohort primary summary are applied to
   the exported per-repeat data; the implemented analyzer emits the unadjusted
   intervals, exact paired tests and operational decisions. Its boolean alone
   never licenses a confirmatory significance claim.
5. Exploratory sensitivity analyses: all-trial STR differences; decided-only
   denominators accompanied by coverage; competence-qualified subsets; old and
   guided prompts; L2; cell-4 raw direction correct/misleading subgroups (with
   each n and L1 recovery); human-coded correction. Consider a logistic model
   with cell, prompt, salience, arm where paired, competence and interactions;
   model identity as a fixed effect, case random intercept for shared cases,
   and repeat fixed effect. Seven selected models cannot support population
   claims about all models. Record convergence issues and do not use exploratory
   model selection to change the confirmatory result.

No weighted headline score combines invariance with sensitivity. A low STR with
high uncertain or wrong-direction rates is not reported as robust performance.

## 8. Offline budget and runtime

Prices checked against OpenRouter's public pages on 2026-10-18. USD per million
input/output tokens; no caching discount, batch discount, tax or credit fee is
assumed. The estimator performs **no API calls**, including no price lookup.

| Frontier candidate | Input / output rate | All 4,380 calls: assumed usage | Initial output-cap scenario | Both cap stages for every slice |
|---|---:|---:|---:|---:|
| [GPT-5.4](https://openrouter.ai/openai/gpt-5.4) | $2.50 / $15 | $165.67 | $596.98 | $1,769.04 |
| [Claude Sonnet 4.6](https://openrouter.ai/anthropic/claude-sonnet-4.6) | $3 / $15 | $170.05 | $601.36 | $1,777.80 |
| [Gemini 3.1 Pro Preview](https://openrouter.ai/google/gemini-3.1-pro-preview) | $2 / $12 | $132.54 | $477.58 | $1,415.23 |
| **All three** | | **$468.26** | **$1,675.92** | **$4,962.07** |

Assume 2,000 input tokens/call, 2,048 output tokens for direct/competence and
4,096 for code, **including reasoning**. Per model: 8.76M input and 9.58464M
assumed output tokens; stage-0 maximum output is 38.33856M. The last column adds
stage 0 plus one full doubled-cap stage 1 (three times stage-0 output and twice
input). These are output-cap scenarios at assumed input length, **not total
billing upper bounds**. Retries can also be billed. Actual usage is reported
from returned token counts; prices may change before collection.

At an assumed 20 output tokens/second locally, 50 through the gateway, 2 seconds
request overhead and 2 seconds/code execution, serial runtime is 135.72 hours
per open model (**542.88 hours for four**) and 55.85 per frontier model
(**167.55 hours for three**). Initial cap scenarios are 536.58 and 217.09 hours
per model, respectively, including the full 20-second execution allowance.
These are arithmetic assumptions, not hardware measurements. Local API cost is
$0; GPU rental, electricity, download/load time, queueing and human annotation
are additional. Open-model compute cost is GPU hourly rate × actual GPU hours.

Reproduce or vary assumptions without network/model access:

```bash
python -m src.estimate_run3 --config configs/run3_open.yaml
python -m src.estimate_run3 --config configs/run3_frontier.yaml
python -m src.estimate_run3 --config configs/run3_frontier.yaml --input-tokens 4000 --direct-output 4096 --code-output 8192
```

## 9. Reproduction and registration checklist

Install Python 3.12 dependencies and regenerate the frozen inputs:

```bash
python -m pip install -r requirements-lock.txt
python -m src.prepare_run3 --config configs/run3_open.yaml
docker build -t mi-run3-sandbox:1 sandbox
python -m pytest
```

Set `MI_RUN_SANDBOX_TESTS=1` to run the Docker integration tests (PowerShell:
`$env:MI_RUN_SANDBOX_TESTS='1'`). The suite also validates old pilot behavior,
rule_v2 branches, classifier fixtures, reasoning payloads and a complete offline
run-3 analysis. CI tests Windows and Linux plus the real Docker boundary.

After external registration, each model/repeat requires these six invocations
of `python -m src.run_pilot`, with common `--config`, `--provider`, `--model`,
`--repeat` and `--results-dir data/results/run3/open` (or `frontier`) arguments:

```text
--dataset competence --arm competence
--dataset pilot --arm direct --variant natural
--dataset pilot --arm direct --variant natural_sustained
--dataset pilot --arm direct --variant guided
--dataset paired --arm direct --variant natural_sustained
--dataset paired --arm code-agent
```

Default run-3 inputs resolve to `data/generated/run3/repeat-N/`; the CLI verifies
their contents. Results automatically go below `repeat-N/stage-0/`. For an
eligible capped slice, repeat its entire invocation with `--cap-stage 1` and
retain stage 0. Do not change the YAML or pass selected case IDs. Each result
has an adjacent immutable manifest with config/dataset hashes, Git commit,
dirty state, package versions and model conditions. Collect from a clean commit.

```bash
python -m src.analyze_run3 --config configs/run3_open.yaml --results-dir data/results/run3/open/repeat-0 --out data/results/run3/open/analysis-repeat-0
```

Repeat analysis for each cohort/repeat. Archive all raw responses (including
thinking where provided), execution records, manifests, failed-attempt audits,
discarded initial capped slices and analysis JSON. Any departure from this
protocol is dated and labelled exploratory; historical run-1/run-2 evidence and
their `rule_v1` interpretations remain frozen.
