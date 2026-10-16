"""Generator tests.

These are the load-bearing tests of the pilot: if the generator mislabels a
case, every downstream number in the paper is wrong and no amount of careful
prompting or grading can recover it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.generate_cases import (
    MeasurementCondition,
    PilotConfig,
    build_case,
    derive_seed,
    generate_dataset,
    load_config,
    measurement_condition_grid,
    read_jsonl,
    write_jsonl,
)
from src.prompts import PromptVariant, build_direct_prompt, render_series_table
from src.schemas import CELL_SPEC, Direction, SalienceLevel, Trend

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg() -> PilotConfig:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def dataset(cfg):
    return generate_dataset(cfg)


def by_cell(dataset, cell):
    return [c for c in dataset if c.cell == cell]


# ------------------------------------------------------------------ 2x2 matrix
def test_dataset_size_and_cell_balance(cfg, dataset):
    assert len(dataset) == cfg.n_per_cell * len(CELL_SPEC)
    for cell in CELL_SPEC:
        assert len(by_cell(dataset, cell)) == cfg.n_per_cell


def test_cell_1_holds_latent_and_measurement_fixed(dataset):
    for case in by_cell(dataset, 1):
        assert case.latent_before == case.latent_after
        assert case.measurement_before.same_function_as(case.measurement_after)
        assert case.measurement_before.same_provenance_as(case.measurement_after)
        assert case.expected_trend is Trend.STABLE
        assert case.true_world_change is False
        assert case.measurement_function_change is False


def test_cell_2_latent_identical_measurement_different(dataset):
    """The main experimental condition: x1 == x2 exactly, f1 != f2."""
    cases = by_cell(dataset, 2)
    assert cases
    for case in cases:
        assert case.latent_before == case.latent_after, "latent must be fixed by construction"
        assert case.latent_shift_ratio == 0.0
        assert not case.measurement_before.same_function_as(case.measurement_after)
        assert case.expected_trend is Trend.STABLE
        assert case.true_world_change is False
        assert case.measurement_function_change is True
        # every timestep carries the identical latent value
        assert len({obs.latent_value for obs in case.observations}) == 1


def test_cell_3_latent_moves_measurement_fixed(dataset):
    for case in by_cell(dataset, 3):
        assert case.latent_before != case.latent_after
        assert case.measurement_before.same_function_as(case.measurement_after)
        assert case.expected_trend is not Trend.STABLE
        assert case.true_world_change is True
        assert case.measurement_function_change is False


def test_cell_4_both_change(dataset):
    for case in by_cell(dataset, 4):
        assert case.latent_before != case.latent_after
        assert not case.measurement_before.same_function_as(case.measurement_after)
        assert case.expected_trend is not Trend.STABLE
        assert case.true_world_change is True
        assert case.measurement_function_change is True


# ---------------------------------------------- ground truth comes from latent
def test_cell_2_ground_truth_ignores_a_large_observed_shift(cfg, dataset):
    """Delta y != 0 while Delta x == 0. This is the whole point of the design."""
    cases = by_cell(dataset, 2)
    for case in cases:
        assert abs(case.observed_shift_ratio) >= cfg.min_observed_shift_ratio
        assert case.expected_trend is Trend.STABLE
    # and the naive read-off of the raw series really does disagree with truth
    fooled = [c for c in cases if c.naive_observed_trend() is not Trend.STABLE]
    assert len(fooled) == len(cases), (
        "every cell-2 case should look like a trend to a provenance-blind reader; "
        "otherwise the condition is not testing anything"
    )


def test_cell_2_generator_never_produces_a_real_latent_trend(dataset):
    for case in by_cell(dataset, 2):
        latents = [obs.latent_value for obs in case.observations]
        assert max(latents) == min(latents)
        assert case.expected_trend is Trend.STABLE


def test_cell_4_truth_can_oppose_the_observed_direction(dataset):
    """At least some cell-4 cases must have observed and latent directions
    disagree -- otherwise cell 4 is solvable by reading the raw series."""
    opposed = [
        c
        for c in by_cell(dataset, 4)
        if c.naive_observed_trend() is not c.expected_trend
    ]
    assert opposed, "cell 4 needs cases where the raw series points the wrong way"


def test_cell_3_truth_is_visible_in_the_raw_series(dataset):
    """Sensitivity control: with the measurement fixed, a provenance-blind read
    of the series should already agree with the latent trend."""
    for case in by_cell(dataset, 3):
        assert case.naive_observed_trend() is case.expected_trend


# ----------------------------------------------------------------- determinism
def test_same_seed_reproduces_identical_cases(cfg):
    a = generate_dataset(cfg)
    b = generate_dataset(cfg)
    assert [c.model_dump(mode="json") for c in a] == [c.model_dump(mode="json") for c in b]


def test_different_master_seed_produces_different_cases(cfg):
    other = cfg.model_copy(update={"master_seed": cfg.master_seed + 1})
    a = generate_dataset(cfg)
    b = generate_dataset(other)
    assert [c.model_dump(mode="json") for c in a] != [c.model_dump(mode="json") for c in b]
    # labels stay balanced even though the numbers differ
    assert [c.cell for c in a] == [c.cell for c in b]


def test_seeds_are_unique_across_the_dataset(dataset):
    seeds = [c.random_seed for c in dataset]
    assert len(set(seeds)) == len(seeds)


def test_case_ids_are_unique(dataset):
    ids = [c.case_id for c in dataset]
    assert len(set(ids)) == len(ids)


def test_derive_seed_is_collision_free():
    seen = {derive_seed(7, cell, i) for cell in CELL_SPEC for i in range(1000)}
    assert len(seen) == len(CELL_SPEC) * 1000


# ---------------------------------------------------------------------- noise
def test_noise_does_not_change_ground_truth(cfg):
    noisy = generate_dataset(cfg)
    clean = generate_dataset(cfg.model_copy(update={"noise_level": 0.0}))
    assert len(noisy) == len(clean)
    for n, c in zip(noisy, clean):
        assert n.case_id == c.case_id
        assert n.expected_trend == c.expected_trend
        assert n.true_world_change == c.true_world_change
        assert n.measurement_function_change == c.measurement_function_change
        assert n.latent_before == c.latent_before
        assert n.latent_after == c.latent_after
        assert n.switch_timestep == c.switch_timestep
        assert n.measurement_after.a == c.measurement_after.a
        assert n.measurement_after.b == c.measurement_after.b
        # only the observed values differ
        assert [o.clean_value for o in n.observations] == [o.clean_value for o in c.observations]


def test_noise_magnitude_is_about_the_configured_level(cfg, dataset):
    residuals = []
    for case in dataset:
        for obs in case.observations:
            if abs(obs.clean_value) > 1e-9:
                residuals.append((obs.observed_value - obs.clean_value) / obs.clean_value)
    mean = sum(residuals) / len(residuals)
    sd = math.sqrt(sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1))
    assert abs(mean) < 0.01
    assert 0.5 * cfg.noise_level < sd < 1.5 * cfg.noise_level


def test_zero_noise_makes_observed_equal_clean(cfg):
    for case in generate_dataset(cfg.model_copy(update={"noise_level": 0.0})):
        for obs in case.observations:
            assert math.isclose(obs.observed_value, obs.clean_value, abs_tol=1e-3)


# ------------------------------------------------------------ design controls
def test_switch_timestep_is_randomised_within_bounds(cfg, dataset):
    lo = max(1, int(cfg.n_timesteps * cfg.switch_min_frac))
    hi = min(cfg.n_timesteps - 1, int(cfg.n_timesteps * cfg.switch_max_frac))
    values = {c.switch_timestep for c in dataset}
    assert all(lo <= v <= hi for v in values)
    assert len(values) > 1, "switch point must vary across cases"


def test_switch_timestep_distribution_does_not_leak_the_cell(dataset):
    """If cell 1 switch points differed from cell 2 switch points, the
    'nothing happened' condition would be identifiable from the layout alone."""
    per_cell = {cell: sorted(c.switch_timestep for c in by_cell(dataset, cell)) for cell in CELL_SPEC}
    means = {cell: sum(v) / len(v) for cell, v in per_cell.items()}
    assert max(means.values()) - min(means.values()) < 3.0


def test_measurement_change_direction_is_balanced(cfg, dataset):
    for cell in (2, 4):
        dirs = [c.measurement_change_direction for c in by_cell(dataset, cell)]
        up = dirs.count(Direction.UP)
        assert abs(up - len(dirs) / 2) <= 1, f"cell {cell} measurement directions unbalanced"


def test_latent_change_direction_is_balanced(cfg, dataset):
    for cell in (3, 4):
        dirs = [c.latent_change_direction for c in by_cell(dataset, cell)]
        up = dirs.count(Direction.UP)
        assert abs(up - len(dirs) / 2) <= 1, f"cell {cell} latent directions unbalanced"


def _grid_counts(dataset, cell):
    """(direction, |gain|, offset) occurrences in one cell."""
    counts = {}
    for case in by_cell(dataset, cell):
        if not case.measurement_function_change:
            continue
        a = case.measurement_after.a
        direction = case.measurement_change_direction
        gain = round(a if direction is Direction.UP else 1.0 / a, 4)
        key = (direction, gain, round(case.measurement_after.b, 4))
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_measurement_grid_is_balanced_within_one(cfg, dataset):
    """Round-robin assignment, not rejection sampling: every grid point occurs
    floor(n/|grid|) or ceil(n/|grid|) times."""
    grid_size = len(measurement_condition_grid(cfg))
    for cell in (2, 4):
        counts = _grid_counts(dataset, cell)
        assert len(counts) == grid_size, f"cell {cell} did not use the whole grid"
        assert max(counts.values()) - min(counts.values()) <= 1


def test_offset_is_not_correlated_with_measurement_direction(cfg, dataset):
    """The bug this replaced: rejection sampling made a positive offset far less
    likely under a downward gain, so 'offset present' partly predicted direction."""
    for cell in (2, 4):
        cases = [c for c in by_cell(dataset, cell) if c.measurement_function_change]
        for offset in cfg.measurement_offsets:
            with_offset = [c for c in cases if round(c.measurement_after.b, 4) == offset]
            assert with_offset, f"offset {offset} never used in cell {cell}"
            up = sum(1 for c in with_offset if c.measurement_change_direction is Direction.UP)
            assert abs(up - len(with_offset) / 2) <= 1, (
                f"cell {cell}: offset {offset} is {up}/{len(with_offset)} upward -- "
                "offset and direction are correlated"
            )


def test_offset_is_not_correlated_with_gain(cfg, dataset):
    for cell in (2, 4):
        counts = _grid_counts(dataset, cell)
        per_pair = {}
        for (_, gain, offset), n in counts.items():
            per_pair[(gain, offset)] = per_pair.get((gain, offset), 0) + n
        expected_pairs = len(cfg.measurement_gains) * len(cfg.measurement_offsets)
        assert len(per_pair) == expected_pairs
        assert max(per_pair.values()) - min(per_pair.values()) <= 1


def test_offset_presence_is_balanced_across_cells(cfg, dataset):
    """Aggregated over both changed-measurement cells, no offset dominates."""
    cases = [c for c in dataset if c.measurement_function_change]
    counts = {
        offset: sum(1 for c in cases if round(c.measurement_after.b, 4) == offset)
        for offset in cfg.measurement_offsets
    }
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_generator_never_rejects_a_draw(cfg):
    """Every grid point must be realisable at every latent level, so that the
    generator is a pure function of its assignment rather than a filter."""
    from src.generate_cases import _baseline_measurement, _build_changed_measurement

    baseline = _baseline_measurement()
    for condition in measurement_condition_grid(cfg):
        for x in cfg.latent_base_range:
            fn = _build_changed_measurement(baseline, condition, x, cfg)
            shift = (fn.apply(x) - baseline.apply(x)) / abs(baseline.apply(x))
            assert abs(shift) >= cfg.min_observed_shift_ratio
            assert (condition.direction is Direction.UP) == (shift > 0)


def test_measurement_direction_matches_observed_shift_sign(dataset):
    for case in by_cell(dataset, 2):
        if case.measurement_change_direction is Direction.UP:
            assert case.observed_shift_ratio > 0
        else:
            assert case.observed_shift_ratio < 0


def test_salience_levels_follow_configured_weights(cfg, dataset):
    for cell in CELL_SPEC:
        cases = by_cell(dataset, cell)
        for level, expected in cfg.salience_weights.items():
            assert sum(1 for c in cases if c.salience_level is level) == expected


def test_only_l1_and_l2_are_generated(dataset):
    assert {c.salience_level for c in dataset} <= {SalienceLevel.L1, SalienceLevel.L2}


def test_latent_changes_exceed_the_noise_floor(cfg, dataset):
    for case in by_cell(dataset, 3) + by_cell(dataset, 4):
        assert abs(case.latent_shift_ratio) >= cfg.latent_rel_delta_range[0] - 1e-9


# ------------------------------------------------------------ config integrity
def test_config_rejects_salience_weights_that_do_not_sum(cfg):
    with pytest.raises(ValueError, match="salience_weights"):
        PilotConfig(**{**cfg.model_dump(), "n_per_cell": 26})


def test_config_rejects_latent_delta_below_noise(cfg):
    with pytest.raises(ValueError, match="exceed the measurement noise"):
        PilotConfig(**{**cfg.model_dump(), "latent_rel_delta_range": (0.01, 0.4)})


def test_build_case_rejects_direction_inconsistent_with_cell(cfg):
    with pytest.raises(ValueError, match="disagree"):
        build_case(
            cell=2,
            index=0,
            salience=SalienceLevel.L1,
            latent_direction=Direction.UP,  # cell 2 must hold latent fixed
            measurement_condition=MeasurementCondition(Direction.UP, 1.2, 0.0),
            cfg=cfg,
        )


def test_build_case_rejects_measurement_condition_inconsistent_with_cell(cfg):
    with pytest.raises(ValueError, match="disagree"):
        build_case(
            cell=1,
            index=0,
            salience=SalienceLevel.L1,
            latent_direction=Direction.NONE,
            measurement_condition=MeasurementCondition(Direction.UP, 1.2, 0.0),
            cfg=cfg,
        )


def test_config_rejects_a_grid_point_below_the_shift_floor(cfg):
    """+-5.0 offsets are exactly what used to force rejection sampling."""
    with pytest.raises(ValueError, match="below min_observed_shift_ratio"):
        PilotConfig(**{**cfg.model_dump(), "measurement_offsets": [0.0, 5.0, -5.0]})


def test_every_configured_grid_point_is_actually_used(cfg, dataset):
    """If the grid were infeasible somewhere the generator would raise, so a
    missing combination means the assignment quietly dropped it."""
    used = {
        (round(c.measurement_after.a, 4), round(c.measurement_after.b, 4))
        for c in dataset
        if c.measurement_function_change
    }
    expected = set()
    for direction in (Direction.UP, Direction.DOWN):
        for gain in cfg.measurement_gains:
            a = round(gain if direction is Direction.UP else 1.0 / gain, 6)
            for offset in cfg.measurement_offsets:
                expected.add((round(a, 4), round(offset, 4)))
    assert used == expected


# --------------------------------------------------------------- serialisation
def test_jsonl_roundtrip(tmp_path, dataset):
    path = write_jsonl(dataset, tmp_path / "cases.jsonl")
    reloaded = read_jsonl(path)
    assert [c.model_dump(mode="json") for c in reloaded] == [
        c.model_dump(mode="json") for c in dataset
    ]
    with open(path, encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.strip()) == len(dataset)


# ------------------------------------------------------------------- prompting
def test_prompt_is_deterministic(dataset):
    case = dataset[0]
    for variant in PromptVariant:
        assert build_direct_prompt(case, variant) == build_direct_prompt(case, variant)


def test_default_variant_is_the_main_arm(dataset):
    case = dataset[0]
    assert build_direct_prompt(case).variant is PromptVariant.NATURAL
    assert build_direct_prompt(case) == build_direct_prompt(case, PromptVariant.NATURAL)


def test_natural_variant_asks_only_for_trend_and_reason(dataset):
    """The main arm must not cue provenance-checking: no measurement flag, no
    mention of instruments, calibration or comparability."""
    for case in dataset[:8]:
        user = build_direct_prompt(case, PromptVariant.NATURAL).user
        task = user.split("Reading log:")[0].lower()
        assert "measurement_change_detected" not in task
        for cue in ("measurement", "calibrat", "instrument change", "comparable", "provenance"):
            assert cue not in task, f"natural prompt cues '{cue}'"
        assert '"trend"' in task and '"reason"' in task


def test_guided_variant_requests_the_measurement_flag(dataset):
    for case in dataset[:8]:
        task = build_direct_prompt(case, PromptVariant.GUIDED).user.split("Reading log:")[0]
        assert "measurement_change_detected" in task


def test_both_variants_show_the_same_log(dataset):
    for case in dataset[:8]:
        natural = build_direct_prompt(case, PromptVariant.NATURAL).user
        guided = build_direct_prompt(case, PromptVariant.GUIDED).user
        assert natural.split("Reading log:")[1] == guided.split("Reading log:")[1]


def test_prompt_contains_every_timestep(dataset):
    for case in dataset[:8]:
        table = render_series_table(case)
        assert len(table.splitlines()) == case.n_timesteps + 2  # header + rule


def test_l1_prompt_exposes_structured_provenance(dataset):
    case = next(c for c in dataset if c.cell == 2 and c.salience_level is SalienceLevel.L1)
    table = render_series_table(case)
    assert "sensor_version" in table and "reference_range" in table
    assert case.measurement_before.sensor_version in table
    assert case.measurement_after.sensor_version in table


def test_l2_prompt_embeds_provenance_and_hides_reference_range(dataset):
    case = next(c for c in dataset if c.cell == 2 and c.salience_level is SalienceLevel.L2)
    table = render_series_table(case)
    assert "device_config_id" in table
    assert "reference_range" not in table
    assert "TX4420-S1-v1" in table and "TX4420-S2-v2" in table


def test_prompt_never_reveals_the_answer(dataset):
    banned = [
        "measurement changed", "assay changed", "calibration changed", "recalibrat",
        "not comparable", "cannot be compared", "no real change", "the temperature rose",
        "sensor was replaced", "ground truth", "stable", "increase", "decrease",
    ]
    for case in dataset:
        for variant in PromptVariant:
            prompt = build_direct_prompt(case, variant)
            # Only the narrative framing is checked; the response schema and the
            # field glossary necessarily name the allowed answer values.
            body = prompt.user.split("Answer with exactly this JSON object")[0].lower()
            for phrase in banned:
                assert phrase not in body, f"{variant.value} prompt leaks '{phrase}'"


def test_prompt_body_is_identical_across_cells(dataset):
    """Everything except the log itself must be constant within a variant, so no
    wording difference can correlate with the condition."""
    for variant in PromptVariant:
        shells = {build_direct_prompt(c, variant).user.split("Reading log:")[0] for c in dataset}
        assert len(shells) == 1, f"{variant.value} prompt varies with the case"
