# Correction labels for run 3

`correction_labels.json` contains 18 deliberately constructed source snippets
with labels assigned from the operational definition before running the screen:
6 positives and 12 negatives. Each includes the reason for its label. They cover
the review's three counterexamples, raw analysis, printing/dropping metadata,
provenance-aware abstention, dead/overwritten correction, normalization, affine
inversion, aliases and control/data flow to the printed `answer`.

These are manually specified synthetic gold labels, not independently annotated
model responses or an estimate of real-world classifier accuracy. Some snippets
assume `df`/`row` inputs; they test static classification, not executability.
`corrected_program.py` is a complete standard-library normalization program;
the Docker integration test executes it against a generated cell-2 CSV and
checks both its output and its correction label. This also covers negative
reference-range lower bounds.

The screen recognizes particular syntactic forms. It does not prove a correct
inverse, calibration choice, cutoff, or program semantics. Helper functions,
dynamic access and hard-coded conversion tables may be missed. Every label is
marked heuristic and requires review. Freeze this fixture set and classifier
before collecting run 3. Later hand coding of model outputs is a separately
reported sensitivity analysis; it must not silently overwrite frozen labels.
