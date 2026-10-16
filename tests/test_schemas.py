"""Schema-level tests: the case object must refuse to hold an inconsistent label."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.schemas import (
    AffineMeasurementFunction,
    BenchmarkCase,
    Direction,
    Domain,
    Observation,
    SalienceLevel,
    Trend,
)

BASE = AffineMeasurementFunction(
    a=1.0, b=0.0, sensor_version="S1", calibration_version="v1", reference_range=(0.0, 100.0)
)
CHANGED = AffineMeasurementFunction(
    a=1.2, b=5.0, sensor_version="S2", calibration_version="v2", reference_range=(5.0, 125.0)
)


def make_case(**overrides):
    """A valid cell-2 case; overrides let each test break exactly one thing."""
    latent_before = overrides.pop("latent_before", 50.0)
    latent_after = overrides.pop("latent_after", 50.0)
    mb = overrides.pop("measurement_before", BASE)
    ma = overrides.pop("measurement_after", CHANGED)
    switch = overrides.pop("switch_timestep", 3)
    n = overrides.pop("n_timesteps", 6)

    observations = []
    for t in range(n):
        segment = "before" if t < switch else "after"
        latent = latent_before if segment == "before" else latent_after
        fn = mb if segment == "before" else ma
        observations.append(
            Observation(
                t=t,
                segment=segment,
                observed_value=fn.apply(latent),
                clean_value=fn.apply(latent),
                latent_value=latent,
            )
        )

    fields = dict(
        case_id="test-001",
        random_seed=1,
        cell=2,
        salience_level=SalienceLevel.L1,
        domain=Domain.INDUSTRIAL_SENSOR,
        n_timesteps=n,
        switch_timestep=switch,
        noise_level=0.0,
        latent_before=latent_before,
        latent_after=latent_after,
        latent_change_direction=Direction.NONE,
        measurement_before=mb,
        measurement_after=ma,
        measurement_change_direction=Direction.UP,
        observations=overrides.pop("observations", observations),
        provenance={"style": "structured_columns"},
        true_world_change=False,
        measurement_function_change=True,
        expected_trend=Trend.STABLE,
        expected_measurement_change_detected=True,
        latent_shift_ratio=0.0,
        observed_shift_ratio=0.3,
    )
    fields.update(overrides)
    return BenchmarkCase(**fields)


# --------------------------------------------------------------- affine algebra
def test_affine_apply_and_invert_roundtrip():
    fn = AffineMeasurementFunction(
        a=1.2, b=5.0, sensor_version="S2", calibration_version="v2", reference_range=(5.0, 125.0)
    )
    assert math.isclose(fn.apply(50.0), 65.0)
    assert math.isclose(fn.invert(65.0), 50.0)
    assert math.isclose(fn.invert(fn.apply(37.3)), 37.3, abs_tol=1e-9)


def test_zero_gain_is_rejected():
    with pytest.raises(ValidationError):
        AffineMeasurementFunction(
            a=0.0, b=1.0, sensor_version="S1", calibration_version="v1",
            reference_range=(0.0, 100.0),
        )


def test_same_function_ignores_provenance_strings():
    twin = AffineMeasurementFunction(
        a=1.0, b=0.0, sensor_version="S9", calibration_version="v9",
        reference_range=(0.0, 100.0),
    )
    assert BASE.same_function_as(twin)
    assert not BASE.same_provenance_as(twin)


# ------------------------------------------------------------------ happy path
def test_valid_case_constructs():
    case = make_case()
    assert case.expected_answer == {"trend": "stable", "measurement_change_detected": True}


# ------------------------------------------------------- label / truth coupling
def test_trend_must_follow_latent_state_not_observations():
    # The observed series clearly rises (1.0*50 -> 1.2*50+5), but latent is fixed.
    with pytest.raises(ValidationError, match="ground truth must be derived from the latent"):
        make_case(expected_trend=Trend.INCREASE)


def test_true_world_change_must_match_latent_values():
    with pytest.raises(ValidationError, match="contradicts the latent state"):
        make_case(true_world_change=True)


def test_measurement_change_flag_must_match_ab():
    with pytest.raises(ValidationError, match="contradicts"):
        make_case(measurement_function_change=False, cell=1)


def test_expected_measurement_change_detected_must_match():
    with pytest.raises(ValidationError, match="expected_measurement_change_detected"):
        make_case(expected_measurement_change_detected=False)


def test_cell_matrix_is_enforced():
    # (latent fixed, measurement changed) is cell 2, never cell 4.
    with pytest.raises(ValidationError, match="requires"):
        make_case(cell=4)


def test_unknown_cell_rejected():
    with pytest.raises(ValidationError):
        make_case(cell=5)


# ------------------------------------------------------------ provenance leaks
def test_changed_provenance_without_changed_function_is_rejected():
    """New version strings on an identical (a, b) would fake an intervention."""
    fake = AffineMeasurementFunction(
        a=1.0, b=0.0, sensor_version="S2", calibration_version="v2",
        reference_range=(0.0, 100.0),
    )
    with pytest.raises(ValidationError, match="must agree"):
        make_case(measurement_after=fake, measurement_function_change=False, cell=1,
                  measurement_change_direction=Direction.NONE)


def test_changed_function_without_changed_provenance_is_rejected():
    """A silent (a, b) change with stale versions would hide the intervention."""
    stealth = AffineMeasurementFunction(
        a=1.2, b=5.0, sensor_version="S1", calibration_version="v1",
        reference_range=(0.0, 100.0),
    )
    with pytest.raises(ValidationError, match="must agree"):
        make_case(measurement_after=stealth)


def test_direction_must_be_none_iff_function_unchanged():
    with pytest.raises(ValidationError, match="direction 'none'"):
        make_case(
            measurement_after=BASE,
            measurement_function_change=False,
            expected_measurement_change_detected=False,
            cell=1,
        )


# ------------------------------------------------------------ series integrity
def test_observation_must_reconstruct_from_latent_and_function():
    case = make_case()
    broken = list(case.observations)
    broken[-1] = Observation(
        t=broken[-1].t, segment="after", observed_value=1.0, clean_value=999.0,
        latent_value=case.latent_after,
    )
    with pytest.raises(ValidationError, match="clean_value"):
        make_case(observations=broken)


def test_segment_labels_must_match_switch_timestep():
    case = make_case()
    broken = list(case.observations)
    first = broken[0]
    broken[0] = Observation(
        t=0, segment="after", observed_value=first.observed_value,
        clean_value=first.clean_value, latent_value=first.latent_value,
    )
    with pytest.raises(ValidationError, match="implies 'before'"):
        make_case(observations=broken)


def test_switch_timestep_must_leave_both_segments_non_empty():
    with pytest.raises(ValidationError, match="switch_timestep"):
        make_case(switch_timestep=0)


def test_observation_count_must_match_n_timesteps():
    case = make_case()
    with pytest.raises(ValidationError, match="expected"):
        make_case(observations=list(case.observations)[:-1])
