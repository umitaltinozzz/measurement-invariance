"""Metrics over graded responses.

Four rules run through this module.

**Every rate carries its own numerator and denominator.**  A rate is not a float;
it is a ``Rate``.  This is what makes a confidence interval computable later
(proje.md section 10 fixes a threshold on Spurious Trend Rate, and a point
estimate over an unstated n cannot be checked against a threshold), and it is
what keeps an empty denominator visible.

**Every critical metric is reported twice.**  A ``DualRate`` holds

* ``valid_only``  -- denominator counts only completions that parsed;
* ``all_trials``  -- denominator counts every completion the model was given,
  with unparseable ones in the denominator and never in the numerator.

An unparseable completion never counts as having solved anything, in either
version.  ``valid_only`` answers "when the model produced an answer, what was it
like"; ``all_trials`` answers "what did asking the model actually get us".
Reporting only ``valid_only`` would let a model improve a score by failing to
answer, which is the exact shape of confound this project studies.

**``valid_only`` is the behavioural metric.**  What the paper claims is about how
a model reasons when it answers, and a diluted denominator does not measure that:
mixing formatting failures into a Spurious Trend Rate makes the number partly a
statement about output formatting.  So the pilot continuation decision reads
``valid_only``.

That would let a model improve a failure rate by declining to answer, so the
defence is a separate gate rather than a diluted denominator: a model x condition
slice whose ``invalid_response_rate`` exceeds
``continuation_criteria.max_invalid_response_rate`` is **not eligible** for the
decision at all.  It is re-run once with structured output.  ``all_trials`` stays
computed and reported as the conservative cross-check, but it is not the primary
failure metric.  See ``src/continuation.py``.

**No headline single score.**  proje.md section 12: a model can score perfectly on
Spurious Trend Rate alone by answering "not comparable" to everything. Spurious
Trend Rate and Miss Rate must always be read together, which is why
``format_metrics_table`` prints them side by side and why no ``overall_score``
exists here.

Nothing in this module assumes the runner used constrained decoding or structured
outputs. If a runner later reduces the invalid rate, that shows up as the two
versions of each metric converging -- it does not change how anything is scored.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from .grader import GradedResponse, ResponseClass
from .prompts import PromptVariant
from .schemas import Trend

#: Fields ``breakdown`` is allowed to group by.
GROUPABLE_FIELDS = ("model", "cell", "salience_level", "prompt_variant")

#: The behavioural basis: what the model does when it answers. Read by the
#: pre-registered kill/continue decision, guarded by the invalid-rate gate.
PRIMARY_BASIS = "valid_only"

#: The conservative cross-check, always reported, never the primary failure metric.
SECONDARY_BASIS = "all_trials"

VALID_BASES = (PRIMARY_BASIS, SECONDARY_BASIS)


class Rate(BaseModel):
    """A proportion that remembers what it was computed from."""

    model_config = ConfigDict(frozen=True)

    numerator: int
    denominator: int
    value: Optional[float] = None

    @classmethod
    def of(cls, numerator: int, denominator: int) -> "Rate":
        if denominator < 0 or numerator < 0:
            raise ValueError("rate components must be non-negative")
        if numerator > denominator:
            raise ValueError(f"numerator {numerator} exceeds denominator {denominator}")
        return cls(
            numerator=numerator,
            denominator=denominator,
            value=(numerator / denominator) if denominator else None,
        )

    @property
    def defined(self) -> bool:
        return self.denominator > 0

    def __str__(self) -> str:
        if not self.defined:
            return "n/a (0)"
        return f"{self.value:.1%} ({self.numerator}/{self.denominator})"


class DualRate(BaseModel):
    """The same count read against two denominators.

    ``valid_only`` excludes unparseable completions from the denominator;
    ``all_trials`` keeps them. The numerator is identical -- an unparseable
    completion is never in it.
    """

    model_config = ConfigDict(frozen=True)

    valid_only: Rate
    all_trials: Rate

    @classmethod
    def of(cls, numerator: int, n_valid: int, n_all: int) -> "DualRate":
        if n_valid > n_all:
            raise ValueError(f"valid count {n_valid} exceeds trial count {n_all}")
        return cls(
            valid_only=Rate.of(numerator, n_valid), all_trials=Rate.of(numerator, n_all)
        )

    def on(self, basis: str = PRIMARY_BASIS) -> Rate:
        if basis not in ("valid_only", "all_trials"):
            raise ValueError(f"basis must be valid_only or all_trials, got {basis!r}")
        return getattr(self, basis)

    @property
    def defined(self) -> bool:
        return self.all_trials.defined

    def __str__(self) -> str:
        return f"all_trials={self.all_trials} | valid_only={self.valid_only}"


def _dual(
    subset: Sequence[GradedResponse], hit: Callable[[GradedResponse], bool]
) -> DualRate:
    """Build a DualRate over a subset defined by CASE properties.

    The subset predicate must depend only on the case and the prompt variant --
    never on whether the completion parsed -- so that the ``all_trials``
    denominator is the number of trials that were actually run.
    """
    valid = [r for r in subset if r.response_class is not ResponseClass.INVALID_RESPONSE]
    numerator = sum(1 for r in valid if hit(r))
    return DualRate.of(numerator, len(valid), len(subset))


def _is(cls: ResponseClass) -> Callable[[GradedResponse], bool]:
    return lambda r: r.response_class is cls


class MetricSet(BaseModel):
    """Every pilot metric for one slice of graded responses."""

    model_config = ConfigDict(frozen=True)

    n_responses: int
    n_valid: int

    #: parse failures over all responses in the slice. Single rate by nature --
    #: its denominator is already every trial. Read this next to every
    #: ``all_trials`` figure below.
    invalid_response_rate: Rate

    #: proje.md 11.1 -- claimed a trend where the latent state never moved.
    #: Subset: responses whose ground truth is `stable` (cells 1, 2).
    spurious_trend_rate: DualRate

    #: proje.md 11.2 -- said `stable` where the latent state really moved.
    #: Subset: responses whose ground truth is a real change (cells 3, 4).
    miss_rate: DualRate

    #: named the opposite direction of a real change; same subset as miss_rate
    wrong_direction_rate: DualRate

    #: abstained where the log did support an answer.
    #: Subset: responses where abstention is NOT warranted -- cell 2 and
    #: cell 4 L2 are excluded, since abstaining there is defensible.
    over_abstention_rate: DualRate

    #: abstained in cell 1, where nothing changed and nothing was hidden
    blanket_skepticism_rate: DualRate

    #: any of correct_stable / correct_change / correct_abstention
    correct_rate: DualRate

    #: guided arm only -- recall of the provenance flag on cases where the
    #: measurement function really did change
    measurement_change_detection_rate: DualRate

    #: guided arm only -- flag raised where the measurement function did not change
    measurement_change_false_alarm_rate: DualRate

    #: proje.md 11.3 -- recovered the latent change through a changed measurement
    #: function. Defined only for cell 4 at L1; L2 x cell 4 is excluded from the
    #: subset entirely because the coefficients are not recoverable there at all.
    latent_recovery_accuracy: DualRate

    class_counts: Dict[str, int]

    @property
    def latent_recovery_defined(self) -> bool:
        return self.latent_recovery_accuracy.defined


def compute_metrics(graded: Sequence[GradedResponse]) -> MetricSet:
    """All metrics for one slice. Slicing is the caller's job -- see ``breakdown``."""
    rows = list(graded)
    n_invalid = sum(1 for r in rows if r.response_class is ResponseClass.INVALID_RESPONSE)

    stable_truth = [r for r in rows if r.expected_trend is Trend.STABLE]
    change_truth = [r for r in rows if r.expected_trend is not Trend.STABLE]
    abstention_unwarranted = [r for r in rows if not r.abstention_warranted]
    cell1 = [r for r in rows if r.cell == 1]

    guided = [r for r in rows if r.prompt_variant is PromptVariant.GUIDED]
    guided_changed = [r for r in guided if r.measurement_function_change]
    guided_unchanged = [r for r in guided if not r.measurement_function_change]

    recovery = [r for r in rows if r.latent_recovery_defined]

    return MetricSet(
        n_responses=len(rows),
        n_valid=len(rows) - n_invalid,
        invalid_response_rate=Rate.of(n_invalid, len(rows)),
        spurious_trend_rate=_dual(stable_truth, _is(ResponseClass.SPURIOUS_TREND)),
        miss_rate=_dual(change_truth, _is(ResponseClass.MISSED_CHANGE)),
        wrong_direction_rate=_dual(change_truth, _is(ResponseClass.WRONG_DIRECTION)),
        over_abstention_rate=_dual(
            abstention_unwarranted, _is(ResponseClass.OVER_ABSTENTION)
        ),
        blanket_skepticism_rate=_dual(cell1, lambda r: r.blanket_skepticism),
        correct_rate=_dual(rows, lambda r: r.is_correct),
        measurement_change_detection_rate=_dual(
            guided_changed, lambda r: r.measurement_change_detected is True
        ),
        measurement_change_false_alarm_rate=_dual(
            guided_unchanged, lambda r: r.measurement_change_detected is True
        ),
        latent_recovery_accuracy=_dual(recovery, lambda r: bool(r.latent_recovery_correct)),
        class_counts={
            cls.value: n
            for cls in ResponseClass
            if (n := sum(1 for r in rows if r.response_class is cls))
        },
    )


def _group_key(row: GradedResponse, by: Sequence[str]) -> Tuple:
    key = []
    for field in by:
        value = getattr(row, field)
        key.append(value.value if hasattr(value, "value") else value)
    return tuple(key)


def breakdown(
    graded: Sequence[GradedResponse], by: Sequence[str] = ("model", "prompt_variant")
) -> Dict[Tuple, MetricSet]:
    """Metrics per group, keyed by the tuple of grouped field values.

    ``by`` may name any of ``GROUPABLE_FIELDS``: model, cell, salience_level,
    prompt_variant.
    """
    unknown = [f for f in by if f not in GROUPABLE_FIELDS]
    if unknown:
        raise ValueError(f"cannot group by {unknown}; allowed: {list(GROUPABLE_FIELDS)}")

    buckets: Dict[Tuple, List[GradedResponse]] = {}
    for row in graded:
        buckets.setdefault(_group_key(row, by), []).append(row)
    return {key: compute_metrics(rows) for key, rows in sorted(buckets.items(), key=repr)}


def class_distribution(graded: Sequence[GradedResponse]) -> Counter:
    return Counter(r.response_class.value for r in graded)


# ------------------------------------------------------------------ reporting
_COLUMNS = [
    ("spurious", "spurious_trend_rate"),
    ("miss", "miss_rate"),
    ("wrong_dir", "wrong_direction_rate"),
    ("over_abst", "over_abstention_rate"),
    ("mc_detect", "measurement_change_detection_rate"),
    ("mc_falarm", "measurement_change_false_alarm_rate"),
    ("latent_rec", "latent_recovery_accuracy"),
]


def _cell(rate: Rate) -> str:
    return "  n/a  " if not rate.defined else f"{rate.value:>6.1%}"


def format_metrics_table(
    results: Dict[Tuple, MetricSet],
    by: Sequence[str] = ("model", "prompt_variant"),
    basis: str = PRIMARY_BASIS,
) -> str:
    """A fixed-width table on one basis.

    Spurious and miss sit next to each other on purpose, and ``invalid`` is always
    the last column: on the ``all_trials`` basis every other figure is attenuated
    by it, so the two must be read together.
    """
    if basis not in ("valid_only", "all_trials"):
        raise ValueError(f"basis must be valid_only or all_trials, got {basis!r}")

    key_width = max(
        [len(" / ".join(str(p) for p in k)) for k in results] + [len(" / ".join(by))]
    )
    header = (
        " / ".join(by).ljust(key_width)
        + " | "
        + " | ".join(name.rjust(6) for name, _ in _COLUMNS)
        + " | invalid |      n"
    )
    lines = [f"basis: {basis}", header, "-" * len(header)]
    for key, metrics in results.items():
        row = " / ".join(str(p) for p in key).ljust(key_width)
        cells = " | ".join(
            _cell(getattr(metrics, attr).on(basis)) for _, attr in _COLUMNS
        )
        lines.append(
            f"{row} | {cells} | {_cell(metrics.invalid_response_rate)} | {metrics.n_responses:>6}"
        )
    return "\n".join(lines)
