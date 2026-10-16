# Pilot results — not run

## Run status

The pilot did not run. Colab reached the Drive mount/bootstrap stage, then displayed a reCAPTCHA service error while connecting the runtime. Following the instruction to stop when a CAPTCHA appears, I did not solve it or reload for a challenge. I declined the pending Drive access prompt and interrupted the queued notebook cells.

The notebook is open in the requested Chrome account, and the Colab runtime type is set to T4 GPU. The bootstrap cell did not complete, so the notebook did not record the actual GPU inventory or a run timestamp.

## Setup and outputs

- Model selection and Q4_K_M quantization are specified in [the Colab config](configs/pilot_colab_t4.yaml). No Ollama model was pulled, and there are no resolved model IDs or weight digests.
- Ollama installation and inference: **not run**.
- Competence pre-test: **not run**; no per-model result rows were produced.
- Direct natural and guided arms: **not run**; no cell-level results were produced.
- Code-agent arm: **not run**. The config keeps generated-code execution disabled, but the notebook stopped before that stage.
- Metrics and continuation criterion: **not run**; there are no runner result files or manifests to verify or score. No continuation verdict is available.
- Cell-by-model chart: **not generated** because there are no observations to plot.
- Results archive download: **not run**. The local project package is present under `data/results/colab_t4/`; it is not a pilot results archive.

No metric values or model performance claims are reported because no runner result files exist. The notebook, provider, analysis path, and local test suite are prepared; a user must clear the Colab reCAPTCHA/runtime blocker before the pilot can start.
