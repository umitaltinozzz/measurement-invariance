"""Schemas for the #108b measurement-invariance pilot benchmark.

Core rule of this project (see proje.md section 3):

    y_t = f_t(x_t)

    Delta y != 0  DOES NOT IMPLY  Delta x != 0

Ground truth is therefore *always* defined on the latent state ``x``.  The
observed series is a derived quantity and never defines truth.  The validators
below enforce that rule structurally, so a buggy generator fails loudly instead
of silently producing a mislabelled dataset.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Absolute tolerance used for all "is this the same latent state / measurement
# function?" comparisons.  Deliberately tight: the generator produces exact
# equality for the "unchanged" conditions, it does not approximate it.
TOL = 1e-9

# Tolerance for reconstructing clean_value = f(x) from stored floats.
RECONSTRUCTION_TOL = 1e-4


class Domain(str, Enum):
    INDUSTRIAL_SENSOR = "industrial_sensor"
    CLINICAL = "clinical"
    FINANCE = "finance"


class SalienceLevel(str, Enum):
    """Provenance salience levels (proje.md section 6).

    L0 (explicit) and L3 (missing) are intentionally not generated in the pilot.
    """

    L1 = "L1"  # implicit structured signal: version / calibration / range columns
    L2 = "L2"  # embedded provenance: identifier that must be parsed


class Trend(str, Enum):
    """Ground-truth trend of the LATENT state, never of the observed series."""

    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


# cell -> (true_world_change, measurement_function_change); proje.md section 4.
CELL_SPEC: Dict[int, Tuple[bool, bool]] = {
    1: (False, False),
    2: (False, True),  # main experimental condition
    3: (True, False),
    4: (True, True),
}


class AffineMeasurementFunction(BaseModel):
    """y = a * x + b, plus the provenance that identifies it.

    The provenance strings are part of the function's identity: two readings
    produced by the same (a, b) must carry the same versions, otherwise the
    dataset would leak a change that did not happen (or hide one that did).
    """

    model_config = ConfigDict(frozen=True)

    a: float
    b: float
    sensor_version: str
    calibration_version: str
    reference_range: Tuple[float, float]

    @field_validator("a")
    @classmethod
    def _gain_must_be_invertible(cls, v: float) -> float:
        if abs(v) < TOL:
            raise ValueError("gain 'a' must be non-zero so the measurement is invertible")
        return v

    @field_validator("reference_range")
    @classmethod
    def _range_must_be_ordered(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"reference_range must be increasing, got {v}")
        return v

    def apply(self, x: float) -> float:
        return self.a * x + self.b

    def invert(self, y: float) -> float:
        return (y - self.b) / self.a

    def same_function_as(self, other: "AffineMeasurementFunction") -> bool:
        """Equality on (a, b) only -- the observable transformation itself."""
        return math.isclose(self.a, other.a, abs_tol=TOL) and math.isclose(
            self.b, other.b, abs_tol=TOL
        )

    def same_provenance_as(self, other: "AffineMeasurementFunction") -> bool:
        return (
            self.sensor_version == other.sensor_version
            and self.calibration_version == other.calibration_version
            and self.reference_range == other.reference_range
        )


class Observation(BaseModel):
    """One timestep.

    Only ``observed_value`` is ever shown to a model.  ``clean_value`` and
    ``latent_value`` are bookkeeping for grading and for the test suite.
    """

    model_config = ConfigDict(frozen=True)

    t: int
    segment: Literal["before", "after"]
    observed_value: float  # noisy f(x); the only field the model sees
    clean_value: float  # noiseless f(x)
    latent_value: float  # hidden ground-truth x


class BenchmarkCase(BaseModel):
    """A single 2x2 benchmark instance.

    Every consistency rule of the design is enforced here rather than in the
    generator, so that a hand-written or mutated case cannot claim a label its
    own numbers contradict.
    """

    model_config = ConfigDict(frozen=True)

    # --- identity / reproducibility -------------------------------------
    case_id: str
    random_seed: int
    cell: int
    salience_level: SalienceLevel
    domain: Domain

    # --- experimental structure ------------------------------------------
    n_timesteps: int
    switch_timestep: int
    noise_level: float

    # --- latent state (ground truth) --------------------------------------
    latent_before: float
    latent_after: float
    latent_change_direction: Direction

    # --- measurement function ---------------------------------------------
    measurement_before: AffineMeasurementFunction
    measurement_after: AffineMeasurementFunction
    measurement_change_direction: Direction

    # --- data --------------------------------------------------------------
    observations: List[Observation]
    provenance: Dict[str, Any]

    # --- labels -------------------------------------------------------------
    true_world_change: bool
    measurement_function_change: bool
    expected_trend: Trend
    expected_measurement_change_detected: bool

    # --- diagnostics (for stratified analysis, not shown to models) ---------
    latent_shift_ratio: float
    observed_shift_ratio: float

    # ---------------------------------------------------------------- checks
    @field_validator("cell")
    @classmethod
    def _cell_range(cls, v: int) -> int:
        if v not in CELL_SPEC:
            raise ValueError(f"cell must be one of {sorted(CELL_SPEC)}, got {v}")
        return v

    @field_validator("noise_level")
    @classmethod
    def _noise_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("noise_level must be >= 0")
        return v

    @model_validator(mode="after")
    def _check_switch_timestep(self) -> "BenchmarkCase":
        if not 0 < self.switch_timestep < self.n_timesteps:
            raise ValueError(
                "switch_timestep must lie strictly inside the series so that both "
                f"segments are non-empty, got {self.switch_timestep} of {self.n_timesteps}"
            )
        return self

    @model_validator(mode="after")
    def _labels_follow_latent_state(self) -> "BenchmarkCase":
        latent_changed = abs(self.latent_after - self.latent_before) > TOL
        if self.true_world_change != latent_changed:
            raise ValueError(
                f"true_world_change={self.true_world_change} contradicts the latent "
                f"state ({self.latent_before} -> {self.latent_after})"
            )

        if latent_changed:
            expected = (
                Trend.INCREASE if self.latent_after > self.latent_before else Trend.DECREASE
            )
        else:
            expected = Trend.STABLE
        if self.expected_trend != expected:
            raise ValueError(
                f"expected_trend={self.expected_trend.value} contradicts the latent state "
                f"({self.latent_before} -> {self.latent_after}); ground truth must be "
                "derived from the latent state, never from observed values"
            )

        expected_dir = {
            Trend.INCREASE: Direction.UP,
            Trend.DECREASE: Direction.DOWN,
            Trend.STABLE: Direction.NONE,
        }[expected]
        if self.latent_change_direction != expected_dir:
            raise ValueError(
                f"latent_change_direction={self.latent_change_direction.value} contradicts "
                f"expected_trend={expected.value}"
            )
        return self

    @model_validator(mode="after")
    def _labels_follow_measurement_function(self) -> "BenchmarkCase":
        fn_changed = not self.measurement_before.same_function_as(self.measurement_after)
        prov_changed = not self.measurement_before.same_provenance_as(self.measurement_after)
        if fn_changed != prov_changed:
            raise ValueError(
                "measurement function change and provenance change must agree: a changed "
                "(a, b) needs new version metadata, and unchanged (a, b) must not carry "
                "new version metadata (that would leak or hide the intervention)"
            )
        if self.measurement_function_change != fn_changed:
            raise ValueError(
                f"measurement_function_change={self.measurement_function_change} contradicts "
                f"(a, b): {self.measurement_before.a, self.measurement_before.b} -> "
                f"{self.measurement_after.a, self.measurement_after.b}"
            )
        if self.expected_measurement_change_detected != fn_changed:
            raise ValueError(
                "expected_measurement_change_detected must equal measurement_function_change"
            )
        if self.measurement_change_direction is Direction.NONE and fn_changed:
            raise ValueError("a changed measurement function needs a signed direction")
        if self.measurement_change_direction is not Direction.NONE and not fn_changed:
            raise ValueError("an unchanged measurement function must have direction 'none'")
        return self

    @model_validator(mode="after")
    def _check_cell_matrix(self) -> "BenchmarkCase":
        expected = CELL_SPEC[self.cell]
        actual = (self.true_world_change, self.measurement_function_change)
        if actual != expected:
            raise ValueError(
                f"cell {self.cell} requires (true_world_change, measurement_function_change)"
                f"={expected}, got {actual}"
            )
        return self

    @model_validator(mode="after")
    def _check_observations(self) -> "BenchmarkCase":
        if len(self.observations) != self.n_timesteps:
            raise ValueError(
                f"expected {self.n_timesteps} observations, got {len(self.observations)}"
            )
        for i, obs in enumerate(self.observations):
            if obs.t != i:
                raise ValueError(f"observations must be contiguous from 0, got t={obs.t} at {i}")
            expected_segment = "before" if i < self.switch_timestep else "after"
            if obs.segment != expected_segment:
                raise ValueError(
                    f"t={i} labelled '{obs.segment}' but switch_timestep="
                    f"{self.switch_timestep} implies '{expected_segment}'"
                )
            latent = self.latent_before if expected_segment == "before" else self.latent_after
            if not math.isclose(obs.latent_value, latent, abs_tol=RECONSTRUCTION_TOL):
                raise ValueError(f"t={i}: latent_value {obs.latent_value} != {latent}")
            fn = (
                self.measurement_before
                if expected_segment == "before"
                else self.measurement_after
            )
            if not math.isclose(obs.clean_value, fn.apply(latent), abs_tol=RECONSTRUCTION_TOL):
                raise ValueError(
                    f"t={i}: clean_value {obs.clean_value} != f(x)={fn.apply(latent)}"
                )
        return self

    # ---------------------------------------------------------------- helpers
    @property
    def expected_answer(self) -> Dict[str, Any]:
        """The grading key for the direct benchmark arm."""
        return {
            "trend": self.expected_trend.value,
            "measurement_change_detected": self.expected_measurement_change_detected,
        }

    def measurement_at(self, t: int) -> AffineMeasurementFunction:
        return self.measurement_before if t < self.switch_timestep else self.measurement_after

    def naive_observed_trend(self) -> Trend:
        """What a model gets if it reads the raw series and ignores provenance.

        Used only by the test suite to prove that ground truth and the naive
        read-off genuinely dissociate in cells 2 and 4.
        """
        before = [o.observed_value for o in self.observations if o.segment == "before"]
        after = [o.observed_value for o in self.observations if o.segment == "after"]
        mb, ma = sum(before) / len(before), sum(after) / len(after)
        if math.isclose(mb, 0.0, abs_tol=TOL):
            return Trend.STABLE
        ratio = (ma - mb) / abs(mb)
        if ratio > self.noise_level or (self.noise_level == 0 and ratio > TOL):
            return Trend.INCREASE
        if ratio < -self.noise_level or (self.noise_level == 0 and ratio < -TOL):
            return Trend.DECREASE
        return Trend.STABLE
