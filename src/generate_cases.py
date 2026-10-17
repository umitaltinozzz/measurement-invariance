"""Deterministic case generator for the #108b pilot benchmark.

Design constraints taken from proje.md:

* Ground truth is computed analytically from the latent state.  No LLM, no
  heuristic, no post-hoc inspection of the observed series.
* Cell 2 (latent fixed, measurement function intervened upon) is the main
  condition; the latent state there is held *exactly* constant by construction.
* Everything is reproducible from a single master seed.

Run:
    python -m src.generate_cases --config configs/pilot.yaml \
        --out data/generated/pilot_cases.jsonl --preview 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas import (
    CELL_SPEC,
    AffineMeasurementFunction,
    BenchmarkCase,
    Direction,
    Domain,
    Observation,
    SalienceLevel,
    Trend,
)

BASELINE_SENSOR_VERSION = "S1"
BASELINE_CALIBRATION_VERSION = "v1"
CHANGED_SENSOR_VERSION = "S2"
CHANGED_CALIBRATION_VERSION = "v2"

# Nominal latent span the sensor is specified over; the reference range shown to
# the model is this span pushed through the measurement function.
LATENT_SPEC_SPAN: Tuple[float, float] = (0.0, 100.0)


class MeasurementCondition(NamedTuple):
    """One point of the measurement-function grid.

    Direction, gain and offset are assigned as a *balanced factorial*, not
    sampled and filtered.  Rejection sampling would silently correlate offset
    with direction (a downward gain plus a positive offset is the combination
    most likely to fall under the minimum-shift threshold), which would make
    "offset present" partly predictable from "readings went down".
    """

    direction: Direction
    gain: float
    offset: float


class ContinuationCriteria(BaseModel):
    """Pre-registered pilot kill/continue thresholds (proje.md section 10).

    Stored as data on purpose: the criterion is fixed *before* the pilot runs, so
    no code change can quietly move it. ``src/continuation.py`` applies these; it
    hardcodes none of them.
    """

    model_config = ConfigDict(frozen=True)

    rule_version: Literal["rule_v1", "rule_v2"] = "rule_v1"
    direct_variant: Literal["natural", "natural_sustained", "guided"] = "natural_sustained"
    direct_salience: SalienceLevel = SalienceLevel.L1
    max_abstention_rate: float = Field(default=0.50, ge=0, le=1)
    min_cases_per_cell: int = Field(default=50, ge=1)

    competence_overall_min: float
    competence_per_skill_min: float
    #: A model x condition slice whose invalid_response_rate exceeds this is not
    #: eligible for the decision. It is re-run once with structured output rather
    #: than being scored through a diluted denominator.
    max_invalid_response_rate: float
    spurious_trend_rate_trigger: float
    provenance_retention_trigger: float

    @model_validator(mode="after")
    def _thresholds_are_proportions(self) -> "ContinuationCriteria":
        for name in (
            "competence_overall_min",
            "competence_per_skill_min",
            "max_invalid_response_rate",
            "spurious_trend_rate_trigger",
            "provenance_retention_trigger",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} must be a proportion in [0, 1]")
        return self


class CompetenceConfig(BaseModel):
    """Thresholds and size of the mandatory competence pre-test (proje.md s.7).

    Data, like ``ContinuationCriteria``: the qualification rule reads these, it
    never hardcodes them.
    """

    model_config = ConfigDict(frozen=True)

    overall_min: float
    per_skill_min: float
    #: A model whose pre-test invalid rate exceeds this is not qualified. Part of
    #: the rule, not a warning: with accuracies read on the valid-only basis, a
    #: model that answers three questions perfectly and garbles the rest would
    #: otherwise clear a gate it has barely been measured against.
    max_invalid_response_rate: float
    cases_per_model: int

    @model_validator(mode="after")
    def _sizes_support_the_planned_split(self) -> "CompetenceConfig":
        if self.cases_per_model % 2:
            raise ValueError(
                f"cases_per_model={self.cases_per_model} must be even to split evenly "
                "between reference-range and affine-inversion"
            )
        half = self.cases_per_model // 2
        if half % 3:
            raise ValueError(
                f"{half} reference-range cases cannot split evenly across "
                "below/within/above; use a count divisible by 6"
            )
        if not 0.0 < self.per_skill_min <= self.overall_min <= 1.0:
            raise ValueError(
                "require 0 < per_skill_min <= overall_min <= 1; a per-skill floor above "
                "the overall floor could never bind"
            )
        return self

    def per_skill_floor_can_bind(self) -> bool:
        """Whether the per-skill floor can ever reject what the overall floor accepts.

        With ``k`` cases per skill and two skills, ``overall >= overall_min``
        allows at most ``w = floor(2k * (1 - overall_min))`` errors. Concentrating
        all of them in one skill gives that skill ``(k - w) / k``. If that is
        still at or above ``per_skill_min``, no admissible error pattern can trip
        the per-skill rule and it is decorative. At 30 cases and 0.90 overall this
        is why ``per_skill_min`` must exceed 0.80.
        """
        k = self.cases_per_model // 2
        max_errors = min(k, int(self.cases_per_model * (1.0 - self.overall_min) + 1e-9))
        worst_skill = (k - max_errors) / k
        return worst_skill < self.per_skill_min


class PairedDatasetConfig(BaseModel):
    """Scope of the paired dataset shared by the direct and code arms.

    proje.md section 8 predicts that a model reasons correctly in prose and then
    loses the same information when it writes code.  That is a *within-case*
    claim, and it can only be tested within a case: if the two arms run on
    different cases, any gap between them is confounded with whatever differs
    between the two case sets.  So both arms answer the same 40 cases, under the
    same ``case_id``, with identical numbers and provenance -- only the
    presentation differs.
    """

    model_config = ConfigDict(frozen=True)

    cells: List[int]
    salience_level: SalienceLevel
    cases_per_cell: int
    case_id_prefix: str = "paired"

    @model_validator(mode="after")
    def _scope_is_sane(self) -> "PairedDatasetConfig":
        if not self.cells:
            raise ValueError("paired.cells must not be empty")
        unknown = [c for c in self.cells if c not in CELL_SPEC]
        if unknown:
            raise ValueError(f"unknown cells {unknown}")
        if self.cases_per_cell < 1:
            raise ValueError("cases_per_cell must be positive")
        if not self.case_id_prefix:
            raise ValueError("case_id_prefix must not be empty")
        return self

    @property
    def total_cases(self) -> int:
        return len(self.cells) * self.cases_per_cell


class CodeAgentConfig(BaseModel):
    """Settings specific to the code arm. Its cases come from ``paired``."""

    model_config = ConfigDict(frozen=True)

    provenance_columns: List[str]
    #: Off by default. See src/sandbox.py: local execution is process isolation,
    #: not a security boundary, and model-written code should run in a container.
    execution_enabled: bool = False
    execution_timeout_seconds: float = 20.0
    protocol_version: Literal["v1", "v2"] = "v1"
    answer_source: Literal["declared", "executed"] = "declared"
    sandbox_image: str = "mi-run3-sandbox:1"
    memory_mb: int = Field(default=512, ge=64)
    cpu_seconds: int = Field(default=10, ge=1)
    pids_limit: int = Field(default=32, ge=8)

    @model_validator(mode="after")
    def _has_provenance_columns(self) -> "CodeAgentConfig":
        if not self.provenance_columns:
            raise ValueError("provenance_columns must not be empty")
        if self.execution_timeout_seconds <= 0:
            raise ValueError("execution timeout must be positive")
        if self.answer_source == "executed" and not self.execution_enabled:
            raise ValueError("executed answers require execution_enabled")
        return self


class ModelSettings(BaseModel):
    """Explicit per-model inference conditions; old configs use runner defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    provider: Literal["ollama", "openrouter"]
    provider_only: Optional[str] = None
    temperature: Optional[float] = 0.0
    send_seed: bool = True
    think: bool | Literal["low", "medium", "high"] | None = None
    reasoning: Optional[Dict[str, Any]] = None
    max_tokens: int = Field(ge=1)
    max_tokens_code_agent: int = Field(ge=1)
    context_length: int = Field(default=65536, ge=1)
    input_usd_per_million: float = Field(default=0, ge=0)
    output_usd_per_million: float = Field(default=0, ge=0)
    tokens_per_second: float = Field(default=20, gt=0)
    request_timeout_seconds: float = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def _budget(self):
        if self.max_tokens_code_agent < self.max_tokens:
            raise ValueError("code token cap must be at least direct cap")
        if self.context_length < self.max_tokens_code_agent + 4096:
            raise ValueError("context must leave at least 4096 input tokens")
        return self


class RunnerConfig(BaseModel):
    """Generation and retry settings. Recorded in every run manifest."""

    model_config = ConfigDict(frozen=True)

    temperature: float
    max_tokens: int
    #: The code arm needs its own ceiling. Measured, not guessed: at 1024 a
    #: reasoning model spends the whole budget thinking and never emits the JSON,
    #: so every completion comes back `finish_reason: length` and is graded
    #: invalid -- a token-budget artefact masquerading as a schema failure.
    max_tokens_code_agent: int
    seed: int
    structured_output: bool = False
    max_transport_retries: int = 4
    backoff_initial_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 30.0
    retry_provider_failures_on_resume: bool = False
    max_provider_resume_attempts: int = Field(default=2, ge=0)
    reject_capped_slices: bool = False

    @model_validator(mode="after")
    def _retry_settings_are_sane(self) -> "RunnerConfig":
        if self.max_tokens < 1 or self.max_tokens_code_agent < 1:
            raise ValueError("token ceilings must be positive")
        if self.max_tokens_code_agent < self.max_tokens:
            raise ValueError(
                "max_tokens_code_agent must be at least max_tokens: the code arm emits "
                "a program plus a justification, so it needs the larger ceiling"
            )
        if self.max_transport_retries < 0:
            raise ValueError("max_transport_retries must be >= 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1")
        if self.backoff_initial_seconds <= 0 or self.backoff_max_seconds <= 0:
            raise ValueError("backoff durations must be positive")
        return self


class PilotConfig(BaseModel):
    """Everything that determines the dataset.  Same config + same seed = same file."""

    model_config = ConfigDict(frozen=True)

    master_seed: int
    domain: Domain = Domain.INDUSTRIAL_SENSOR
    n_per_cell: int
    n_timesteps: int
    noise_level: float
    switch_min_frac: float
    switch_max_frac: float
    salience_weights: Dict[SalienceLevel, int]
    latent_base_range: Tuple[float, float]
    latent_rel_delta_range: Tuple[float, float]
    measurement_gains: List[float]
    measurement_offsets: List[float]
    min_observed_shift_ratio: float
    #: Overrides the case_id prefix. None keeps the domain name, so the original
    #: pilot dataset regenerates byte-identically.
    case_id_prefix: Optional[str] = None
    continuation_criteria: ContinuationCriteria
    competence: CompetenceConfig
    paired: PairedDatasetConfig
    code_agent: CodeAgentConfig
    runner: RunnerConfig
    models: Dict[str, ModelSettings] = Field(default_factory=dict)
    repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _run3_conditions(self):
        if self.continuation_criteria.rule_version != "rule_v2":
            return self
        if not self.models:
            raise ValueError("rule_v2 requires explicit per-model settings")
        for name, setting in self.models.items():
            if setting.provider == "ollama" and setting.think is None:
                raise ValueError("Ollama reasoning must be explicit")
            if name.startswith("gpt-oss") and setting.think not in ("low", "medium", "high"):
                raise ValueError("gpt-oss requires a named reasoning level")
            if name.startswith("qwen3") and type(setting.think) is not bool:
                raise ValueError("qwen3 requires an explicit boolean think setting")
            if setting.provider == "openrouter" and setting.reasoning is None:
                raise ValueError("OpenRouter reasoning must be explicit")
            if setting.provider == "openrouter" and not setting.provider_only:
                raise ValueError("OpenRouter upstream must be pinned")
        if self.code_agent.protocol_version != "v2":
            raise ValueError("rule_v2 requires code protocol v2")
        return self

    @model_validator(mode="after")
    def _competence_floors_match_the_continuation_criteria(self) -> "PilotConfig":
        pairs = (
            ("overall_min", self.competence.overall_min,
             "competence_overall_min", self.continuation_criteria.competence_overall_min),
            ("per_skill_min", self.competence.per_skill_min,
             "competence_per_skill_min", self.continuation_criteria.competence_per_skill_min),
            ("max_invalid_response_rate", self.competence.max_invalid_response_rate,
             "max_invalid_response_rate", self.continuation_criteria.max_invalid_response_rate),
        )
        for own_name, own, other_name, other in pairs:
            if own != other:
                raise ValueError(
                    f"competence.{own_name}={own} must equal "
                    f"continuation_criteria.{other_name}={other}; proje.md section 10 fixes "
                    "one set of competence floors and the two must not drift apart"
                )
        return self

    @field_validator("measurement_gains")
    @classmethod
    def _gains_above_one(cls, v: List[float]) -> List[float]:
        # Directions are produced by using g or 1/g, so the catalogue is > 1.
        if any(g <= 1.0 for g in v):
            raise ValueError("measurement_gains must all be > 1.0; direction is applied later")
        return v

    @model_validator(mode="after")
    def _weights_match_cell_size(self) -> "PilotConfig":
        total = sum(self.salience_weights.values())
        if total != self.n_per_cell:
            raise ValueError(
                f"salience_weights sum to {total} but n_per_cell is {self.n_per_cell}"
            )
        if not 0.0 < self.switch_min_frac <= self.switch_max_frac < 1.0:
            raise ValueError("require 0 < switch_min_frac <= switch_max_frac < 1")
        lo, hi = self.latent_rel_delta_range
        if not 0.0 < lo <= hi:
            raise ValueError("latent_rel_delta_range must be positive and increasing")
        if lo <= self.noise_level:
            raise ValueError(
                "smallest latent change must exceed the measurement noise level, otherwise "
                "cell 3/4 ground truth is not recoverable even in principle"
            )
        return self

    @model_validator(mode="after")
    def _measurement_grid_is_feasible_everywhere(self) -> "PilotConfig":
        """Every (direction, gain, offset) cell must be usable at every latent level.

        The relative shift induced by y = a*x + b against the identity baseline is

            r(x) = (a - 1) + b / x

        which is monotone in x, so checking both ends of ``latent_base_range``
        proves the whole interval.  If every grid point is feasible everywhere,
        the generator never has to reject a draw, and direction, gain and offset
        stay mutually independent by construction.
        """
        for direction in (Direction.UP, Direction.DOWN):
            for gain in self.measurement_gains:
                a = gain if direction is Direction.UP else 1.0 / gain
                for offset in self.measurement_offsets:
                    for x in self.latent_base_range:
                        r = (a - 1.0) + offset / x
                        if abs(r) < self.min_observed_shift_ratio:
                            raise ValueError(
                                f"measurement grid point (direction={direction.value}, "
                                f"gain={gain}, offset={offset}) induces only {r:+.3f} "
                                f"relative shift at x={x}, below min_observed_shift_ratio="
                                f"{self.min_observed_shift_ratio}; shrink the offsets or "
                                "widen the gains rather than filtering draws"
                            )
                        if (direction is Direction.UP) != (r > 0):
                            raise ValueError(
                                f"measurement grid point (direction={direction.value}, "
                                f"gain={gain}, offset={offset}) shifts the readings the "
                                f"wrong way at x={x} (r={r:+.3f})"
                            )
        return self


def load_config(path: str | Path) -> PilotConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PilotConfig(**raw)


def derive_seed(master_seed: int, cell: int, index: int) -> int:
    """Stable, collision-free per-case seed. No hashing (hash() is salted)."""
    return master_seed * 1_000_000 + cell * 10_000 + index


def _balanced(values: Sequence[Any], n: int, rng: random.Random) -> List[Any]:
    """A length-n list using `values` as evenly as possible, then shuffled."""
    out = [values[i % len(values)] for i in range(n)]
    rng.shuffle(out)
    return out


def _baseline_measurement() -> AffineMeasurementFunction:
    return AffineMeasurementFunction(
        a=1.0,
        b=0.0,
        sensor_version=BASELINE_SENSOR_VERSION,
        calibration_version=BASELINE_CALIBRATION_VERSION,
        reference_range=LATENT_SPEC_SPAN,
    )


NO_MEASUREMENT_CHANGE = MeasurementCondition(Direction.NONE, 1.0, 0.0)


def measurement_condition_grid(cfg: PilotConfig) -> List[MeasurementCondition]:
    """The full direction x gain x offset factorial, in a fixed order."""
    return [
        MeasurementCondition(direction, gain, offset)
        for direction in (Direction.UP, Direction.DOWN)
        for gain in cfg.measurement_gains
        for offset in cfg.measurement_offsets
    ]


def _measurement_condition_pool(
    cfg: PilotConfig, n: int, rotation: int, rng: random.Random
) -> List[MeasurementCondition]:
    """`n` conditions drawn round-robin from the factorial grid, then shuffled.

    Round-robin (rather than independent sampling) makes every grid point occur
    floor(n/|grid|) or ceil(n/|grid|) times, so direction, gain and offset are
    balanced and mutually uncorrelated up to that +-1.  `rotation` moves which
    grid points get the remainder, so the leftover does not always land on the
    same combination across cells.
    """
    grid = measurement_condition_grid(cfg)
    pool = [grid[(rotation + i) % len(grid)] for i in range(n)]
    rng.shuffle(pool)
    return pool


def _build_changed_measurement(
    baseline: AffineMeasurementFunction,
    condition: MeasurementCondition,
    latent_reference_point: float,
    cfg: PilotConfig,
) -> AffineMeasurementFunction:
    """Realise one grid point as a concrete measurement function.

    Deterministic: no draw, no retry.  The grid is proven feasible by
    ``PilotConfig._measurement_grid_is_feasible_everywhere``; the check here
    guards against that proof and this construction drifting apart.
    """
    a2 = round(
        condition.gain if condition.direction is Direction.UP else 1.0 / condition.gain, 6
    )
    # The offset is NOT flipped with direction -- flipping it would reintroduce
    # exactly the offset/direction correlation this grid exists to remove.
    b2 = round(condition.offset, 6)

    y_before = baseline.apply(latent_reference_point)
    shift = (a2 * latent_reference_point + b2 - y_before) / abs(y_before)
    if abs(shift) < cfg.min_observed_shift_ratio or (
        (condition.direction is Direction.UP) != (shift > 0)
    ):
        raise RuntimeError(
            f"grid point {condition} yields shift {shift:+.4f} at x="
            f"{latent_reference_point}, violating direction/min_observed_shift_ratio"
        )

    lo = a2 * LATENT_SPEC_SPAN[0] + b2
    hi = a2 * LATENT_SPEC_SPAN[1] + b2
    return AffineMeasurementFunction(
        a=a2,
        b=b2,
        sensor_version=CHANGED_SENSOR_VERSION,
        calibration_version=CHANGED_CALIBRATION_VERSION,
        reference_range=(round(min(lo, hi), 2), round(max(lo, hi), 2)),
    )


def _provenance_spec(level: SalienceLevel) -> Dict[str, Any]:
    """Declares *how* provenance is exposed; the values live on the measurement
    functions.  prompts.py renders from this pair."""
    if level is SalienceLevel.L1:
        return {
            "style": "structured_columns",
            "fields": ["sensor_version", "calibration_version", "reference_range"],
        }
    return {
        "style": "embedded_identifier",
        "fields": ["device_config_id"],
        "template": "TX4420-{sensor_version}-{calibration_version}",
    }


def build_case(
    cell: int,
    index: int,
    salience: SalienceLevel,
    latent_direction: Direction,
    measurement_condition: MeasurementCondition,
    cfg: PilotConfig,
) -> BenchmarkCase:
    latent_changes, measurement_changes = CELL_SPEC[cell]
    measurement_direction = measurement_condition.direction
    if latent_changes != (latent_direction is not Direction.NONE):
        raise ValueError(f"cell {cell} and latent_direction {latent_direction} disagree")
    if measurement_changes != (measurement_direction is not Direction.NONE):
        raise ValueError(f"cell {cell} and measurement_condition disagree")

    seed = derive_seed(cfg.master_seed, cell, index)
    rng = random.Random(seed)

    # --- latent state -----------------------------------------------------
    latent_before = round(rng.uniform(*cfg.latent_base_range), 4)
    if latent_direction is Direction.NONE:
        # Held fixed *by construction*, not by rounding. This is the core of cell 2.
        latent_after = latent_before
        expected_trend = Trend.STABLE
    else:
        delta = rng.uniform(*cfg.latent_rel_delta_range)
        sign = 1.0 if latent_direction is Direction.UP else -1.0
        latent_after = round(latent_before * (1.0 + sign * delta), 4)
        expected_trend = Trend.INCREASE if sign > 0 else Trend.DECREASE

    # --- measurement function ---------------------------------------------
    measurement_before = _baseline_measurement()
    if measurement_direction is Direction.NONE:
        measurement_after = measurement_before
    else:
        measurement_after = _build_changed_measurement(
            measurement_before, measurement_condition, latent_before, cfg
        )

    # --- switch point ------------------------------------------------------
    lo_t = max(1, int(cfg.n_timesteps * cfg.switch_min_frac))
    hi_t = min(cfg.n_timesteps - 1, int(cfg.n_timesteps * cfg.switch_max_frac))
    switch_timestep = rng.randint(lo_t, hi_t)

    # --- observed series ---------------------------------------------------
    observations: List[Observation] = []
    for t in range(cfg.n_timesteps):
        segment = "before" if t < switch_timestep else "after"
        latent = latent_before if segment == "before" else latent_after
        fn = measurement_before if segment == "before" else measurement_after
        clean = fn.apply(latent)
        # gauss() is drawn unconditionally so that the random stream -- and hence
        # every other property of the case -- is independent of noise_level.
        noise_draw = rng.gauss(0.0, 1.0)
        observed = clean * (1.0 + cfg.noise_level * noise_draw)
        observations.append(
            Observation(
                t=t,
                segment=segment,
                observed_value=round(observed, 4),
                clean_value=round(clean, 6),
                latent_value=latent,
            )
        )

    clean_before_level = measurement_before.apply(latent_before)
    clean_after_level = measurement_after.apply(latent_after)

    return BenchmarkCase(
        case_id=f"{cfg.case_id_prefix or cfg.domain.value}-c{cell}-{salience.value}-{index:03d}",
        random_seed=seed,
        cell=cell,
        salience_level=salience,
        domain=cfg.domain,
        n_timesteps=cfg.n_timesteps,
        switch_timestep=switch_timestep,
        noise_level=cfg.noise_level,
        latent_before=latent_before,
        latent_after=latent_after,
        latent_change_direction=latent_direction,
        measurement_before=measurement_before,
        measurement_after=measurement_after,
        measurement_change_direction=measurement_direction,
        observations=observations,
        provenance=_provenance_spec(salience),
        true_world_change=latent_changes,
        measurement_function_change=measurement_changes,
        expected_trend=expected_trend,
        expected_measurement_change_detected=measurement_changes,
        latent_shift_ratio=round((latent_after - latent_before) / abs(latent_before), 6),
        observed_shift_ratio=round(
            (clean_after_level - clean_before_level) / abs(clean_before_level), 6
        ),
    )


def generate_dataset(cfg: PilotConfig) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    for cell in sorted(CELL_SPEC):
        latent_changes, measurement_changes = CELL_SPEC[cell]
        # A dedicated RNG per cell for the *assignment* of conditions, so that
        # changing n_per_cell in one cell cannot perturb another.
        assign_rng = random.Random(derive_seed(cfg.master_seed, cell, 999_999))

        salience_pool: List[SalienceLevel] = []
        for level, count in sorted(cfg.salience_weights.items(), key=lambda kv: kv[0].value):
            salience_pool.extend([level] * count)
        assign_rng.shuffle(salience_pool)

        latent_dirs = (
            _balanced([Direction.UP, Direction.DOWN], cfg.n_per_cell, assign_rng)
            if latent_changes
            else [Direction.NONE] * cfg.n_per_cell
        )
        measurement_conditions = (
            _measurement_condition_pool(cfg, cfg.n_per_cell, rotation=cell, rng=assign_rng)
            if measurement_changes
            else [NO_MEASUREMENT_CHANGE] * cfg.n_per_cell
        )

        for index in range(cfg.n_per_cell):
            cases.append(
                build_case(
                    cell=cell,
                    index=index,
                    salience=salience_pool[index],
                    latent_direction=latent_dirs[index],
                    measurement_condition=measurement_conditions[index],
                    cfg=cfg,
                )
            )
    return cases


def write_jsonl(cases: Sequence[BenchmarkCase], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: str | Path) -> List[BenchmarkCase]:
    with open(path, "r", encoding="utf-8") as fh:
        return [BenchmarkCase(**json.loads(line)) for line in fh if line.strip()]


def format_case(case: BenchmarkCase, max_rows: int = 8) -> str:
    """Human-readable dump, ground truth included. Never shown to a model."""
    lines = [
        f"case_id                 : {case.case_id}",
        f"cell / salience / seed  : {case.cell} / {case.salience_level.value} / {case.random_seed}",
        f"latent  x1 -> x2        : {case.latent_before} -> {case.latent_after} "
        f"({case.latent_shift_ratio:+.1%})",
        f"measure f1              : y = {case.measurement_before.a}*x + {case.measurement_before.b} "
        f"[{case.measurement_before.sensor_version}/"
        f"{case.measurement_before.calibration_version}, "
        f"range {list(case.measurement_before.reference_range)}]",
        f"measure f2              : y = {case.measurement_after.a}*x + {case.measurement_after.b} "
        f"[{case.measurement_after.sensor_version}/"
        f"{case.measurement_after.calibration_version}, "
        f"range {list(case.measurement_after.reference_range)}]",
        f"switch timestep         : {case.switch_timestep} of {case.n_timesteps}",
        f"observed level shift    : {case.observed_shift_ratio:+.1%}",
        f"GROUND TRUTH            : trend={case.expected_trend.value}, "
        f"true_world_change={case.true_world_change}, "
        f"measurement_change={case.measurement_function_change}",
        f"naive read of raw series: {case.naive_observed_trend().value}",
        "series (t, observed, [segment]):",
    ]
    head = case.observations[: max_rows // 2]
    tail = case.observations[-(max_rows // 2) :]
    for obs in head:
        lines.append(f"    t={obs.t:>2}  {obs.observed_value:>9.3f}  [{obs.segment}]")
    lines.append("    ...")
    for obs in tail:
        lines.append(f"    t={obs.t:>2}  {obs.observed_value:>9.3f}  [{obs.segment}]")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the #108b pilot dataset.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--out", default="data/generated/pilot_cases.jsonl")
    parser.add_argument("--preview", type=int, default=4, help="cases to print after writing")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cases = generate_dataset(cfg)
    out = write_jsonl(cases, args.out)

    counts: Dict[int, int] = {}
    for case in cases:
        counts[case.cell] = counts.get(case.cell, 0) + 1
    print(f"wrote {len(cases)} cases -> {out}")
    print("cell distribution: " + ", ".join(f"cell {c}: {n}" for c, n in sorted(counts.items())))

    for case in cases[: args.preview]:
        print("\n" + "-" * 78)
        print(format_case(case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
