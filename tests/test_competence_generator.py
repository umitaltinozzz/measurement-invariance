"""Competence generator tests.

The pre-test gates which models may be interpreted at all, so a mislabelled
competence case would silently change the paper's population.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.competence_prompts import (
    build_affine_prompt,
    build_competence_prompt,
    build_reference_range_prompt,
)
from src.competence_schemas import (
    AffineInversionCase,
    BoundaryKind,
    CompetenceSkill,
    RangeClass,
    ReferenceRangeCase,
    classify_against_range,
    numeric_tolerance,
    within_tolerance,
)
from src.generate_cases import load_config
from src.generate_competence import (
    ANALYTES,
    build_affine_case,
    build_reference_range_case,
    derive_competence_seed,
    generate_competence_dataset,
    read_competence_jsonl,
    write_competence_jsonl,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cases(cfg):
    return generate_competence_dataset(cfg)


def rr(cases):
    return [c for c in cases if c.skill is CompetenceSkill.REFERENCE_RANGE]


def af(cases):
    return [c for c in cases if c.skill is CompetenceSkill.AFFINE_INVERSION]


# ============================================================ composition (1-3)
def test_thirty_cases(cfg, cases):
    assert cfg.competence.cases_per_model == 30
    assert len(cases) == 30


def test_fifteen_of_each_skill(cases):
    assert len(rr(cases)) == 15
    assert len(af(cases)) == 15


def test_reference_range_classes_are_five_each(cases):
    counts = {cls: 0 for cls in RangeClass}
    for case in rr(cases):
        counts[case.expected_classification] += 1
    assert counts == {RangeClass.BELOW: 5, RangeClass.WITHIN: 5, RangeClass.ABOVE: 5}


def test_case_ids_and_seeds_are_unique(cases):
    assert len({c.case_id for c in cases}) == len(cases)
    assert len({c.random_seed for c in cases}) == len(cases)


def test_seed_derivation_is_collision_free():
    seeds = {derive_competence_seed(20260824, i) for i in range(5000)}
    assert len(seeds) == 5000


def test_competence_seeds_never_collide_with_benchmark_seeds(cfg):
    from src.generate_cases import derive_seed
    from src.schemas import CELL_SPEC

    benchmark = {
        derive_seed(cfg.master_seed, cell, i) for cell in CELL_SPEC for i in range(1000)
    }
    competence = {derive_competence_seed(cfg.master_seed, i) for i in range(1000)}
    assert not benchmark & competence


# ================================================================ ground truth (4)
def test_reference_range_labels_match_the_closed_interval(cases):
    for case in rr(cases):
        assert case.expected_classification is classify_against_range(
            case.result, case.reference_range
        )


def test_affine_answers_satisfy_their_equation(cases):
    for case in af(cases):
        assert within_tolerance(case.a * case.expected_x + case.b, case.y)


def test_boundary_cases_are_present_and_classified_as_within(cases):
    kinds = {c.boundary_kind for c in rr(cases)}
    assert BoundaryKind.AT_LOWER in kinds and BoundaryKind.AT_UPPER in kinds
    for case in rr(cases):
        if case.boundary_kind is BoundaryKind.AT_LOWER:
            assert case.result == case.reference_range[0]
            assert case.expected_classification is RangeClass.WITHIN
        if case.boundary_kind is BoundaryKind.AT_UPPER:
            assert case.result == case.reference_range[1]
            assert case.expected_classification is RangeClass.WITHIN


def test_near_boundary_cases_land_outside(cases):
    for case in rr(cases):
        if case.boundary_kind is BoundaryKind.JUST_BELOW:
            assert case.expected_classification is RangeClass.BELOW
            assert case.result < case.reference_range[0]
        if case.boundary_kind is BoundaryKind.JUST_ABOVE:
            assert case.expected_classification is RangeClass.ABOVE
            assert case.result > case.reference_range[1]


def test_boundary_cases_sit_inside_the_planned_class_counts(cases):
    """Boundary cases are placed into the 5/5/5 split, not added on top of it."""
    boundary = [c for c in rr(cases) if c.boundary_kind is not BoundaryKind.NONE]
    assert len(boundary) == 4  # at_lower, at_upper, just_below, just_above
    assert len(rr(cases)) == 15


def test_non_boundary_results_never_sit_on_an_endpoint(cases):
    for case in rr(cases):
        if case.boundary_kind is BoundaryKind.NONE:
            assert case.result not in case.reference_range


# ================================================================== variety
def test_ranges_include_negatives_and_decimals(cases):
    lows = [c.reference_range[0] for c in rr(cases)]
    assert any(low < 0 for low in lows)
    assert any(not float(c.reference_range[1]).is_integer() for c in rr(cases))


def test_range_spans_vary_by_orders_of_magnitude(cases):
    spans = [c.reference_range[1] - c.reference_range[0] for c in rr(cases)]
    assert max(spans) / min(spans) > 100


def test_affine_uses_positive_gains_only(cases):
    assert all(c.a > 0 for c in af(cases))


def test_affine_offsets_include_both_signs(cases):
    offsets = [c.b for c in af(cases)]
    assert any(b > 0 for b in offsets) and any(b < 0 for b in offsets)


def test_affine_answers_include_integers_and_decimals(cases):
    xs = [c.expected_x for c in af(cases)]
    assert any(float(x).is_integer() for x in xs)
    assert any(not float(x).is_integer() for x in xs)


def test_affine_is_not_only_the_benchmark_gains(cases):
    """The skill must transfer, not be memorised from the two pilot gains."""
    gains = {c.a for c in af(cases)}
    assert {1.2, 2.0} <= gains
    assert len(gains - {1.2, 2.0}) >= 4


def test_affine_values_are_numerically_stable(cases):
    for case in af(cases):
        assert case.a >= 0.2
        recovered = (case.y - case.b) / case.a
        assert abs(recovered - case.expected_x) < numeric_tolerance(case.expected_x)


def test_analytes_are_fictional(cases):
    for case in rr(cases):
        assert case.analyte in ANALYTES
        assert case.analyte.split("-")[0] in {"Analyte", "Marker", "Index", "Compound"}


# ================================================================ determinism (5)
def test_same_seed_reproduces_identical_cases(cfg):
    a = generate_competence_dataset(cfg)
    b = generate_competence_dataset(cfg)
    assert [c.model_dump(mode="json") for c in a] == [c.model_dump(mode="json") for c in b]


def test_different_master_seed_changes_the_dataset(cfg):
    other = cfg.model_copy(update={"master_seed": cfg.master_seed + 1})
    a = generate_competence_dataset(cfg)
    b = generate_competence_dataset(other)
    assert [c.model_dump(mode="json") for c in a] != [c.model_dump(mode="json") for c in b]
    # the plan is unchanged; only the numbers move
    assert [c.skill for c in a] == [c.skill for c in b]


def test_jsonl_roundtrip_is_byte_identical(tmp_path, cases):
    first = write_competence_jsonl(cases, tmp_path / "a.jsonl")
    reloaded = read_competence_jsonl(first)
    second = write_competence_jsonl(reloaded, tmp_path / "b.jsonl")
    assert first.read_bytes() == second.read_bytes()
    assert [c.model_dump(mode="json") for c in reloaded] == [
        c.model_dump(mode="json") for c in cases
    ]


def test_jsonl_roundtrip_preserves_case_types(tmp_path, cases):
    reloaded = read_competence_jsonl(write_competence_jsonl(cases, tmp_path / "c.jsonl"))
    assert [type(c) for c in reloaded] == [type(c) for c in cases]


# ============================================================ schema enforcement
class TestSchemaRefusesBadLabels:
    def test_wrong_range_classification_is_rejected(self):
        with pytest.raises(ValidationError, match="contradicts"):
            ReferenceRangeCase(
                case_id="x", random_seed=1, analyte="Analyte-X17", result=120.0,
                reference_range=(0.0, 150.0), expected_classification=RangeClass.ABOVE,
            )

    def test_endpoint_counts_as_within(self):
        case = ReferenceRangeCase(
            case_id="x", random_seed=1, analyte="Analyte-X17", result=150.0,
            reference_range=(0.0, 150.0), expected_classification=RangeClass.WITHIN,
            boundary_kind=BoundaryKind.AT_UPPER,
        )
        assert case.expected_classification is RangeClass.WITHIN

    def test_endpoint_labelled_above_is_rejected(self):
        with pytest.raises(ValidationError, match="contradicts"):
            ReferenceRangeCase(
                case_id="x", random_seed=1, analyte="Analyte-X17", result=150.0,
                reference_range=(0.0, 150.0), expected_classification=RangeClass.ABOVE,
                boundary_kind=BoundaryKind.AT_UPPER,
            )

    def test_real_analyte_names_are_rejected(self):
        with pytest.raises(ValidationError, match="fictional"):
            ReferenceRangeCase(
                case_id="x", random_seed=1, analyte="Ferritin", result=120.0,
                reference_range=(0.0, 150.0), expected_classification=RangeClass.WITHIN,
            )

    def test_unmarked_endpoint_result_is_rejected(self):
        with pytest.raises(ValidationError, match="endpoint"):
            ReferenceRangeCase(
                case_id="x", random_seed=1, analyte="Analyte-X17", result=0.0,
                reference_range=(0.0, 150.0), expected_classification=RangeClass.WITHIN,
                boundary_kind=BoundaryKind.NONE,
            )

    def test_inverted_range_is_rejected(self):
        with pytest.raises(ValidationError, match="increasing"):
            ReferenceRangeCase(
                case_id="x", random_seed=1, analyte="Analyte-X17", result=1.0,
                reference_range=(150.0, 0.0), expected_classification=RangeClass.WITHIN,
            )

    def test_affine_answer_that_does_not_solve_the_equation_is_rejected(self):
        with pytest.raises(ValidationError, match="does not satisfy"):
            AffineInversionCase(case_id="x", random_seed=1, a=1.2, b=5.0, y=65.0, expected_x=49.0)

    def test_affine_accepts_the_worked_example_from_proje_md(self):
        case = AffineInversionCase(case_id="x", random_seed=1, a=1.2, b=5.0, y=65.0, expected_x=50.0)
        assert case.expected_answer == 50.0

    def test_non_positive_gain_is_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            AffineInversionCase(case_id="x", random_seed=1, a=-1.2, b=5.0, y=-55.0, expected_x=50.0)

    def test_ill_conditioned_gain_is_rejected(self):
        with pytest.raises(ValidationError, match="below"):
            AffineInversionCase(case_id="x", random_seed=1, a=0.01, b=0.0, y=0.5, expected_x=50.0)


# ============================================================ tolerance (6, 7)
class TestNumericTolerance:
    def test_formula_matches_the_specification(self):
        assert numeric_tolerance(50.0) == pytest.approx(max(1e-6, 50.0 * 1e-4))
        assert numeric_tolerance(0.0) == 1e-6
        assert numeric_tolerance(-8.0) == pytest.approx(8.0 * 1e-4)

    def test_accepts_a_value_inside_the_band(self):
        assert within_tolerance(50.0049, 50.0)
        assert within_tolerance(50.0, 50.0)
        assert within_tolerance(50, 50.0)

    def test_rejects_a_value_just_outside_the_band(self):
        assert not within_tolerance(50.0 + numeric_tolerance(50.0) * 1.01, 50.0)
        assert not within_tolerance(49.0, 50.0)

    def test_rejects_non_finite_answers(self):
        assert not within_tolerance(float("nan"), 50.0)
        assert not within_tolerance(float("inf"), 50.0)

    def test_absolute_floor_protects_near_zero(self):
        assert within_tolerance(1e-7, 0.0)
        assert not within_tolerance(1e-3, 0.0)


# ================================================================ prompt leaks
BANNED = [
    "measurement invariance", "measurement", "invarian", "world change",
    "instrument", "spurious", "trend", "provenance", "calibrat", "sensor",
    "assay", "comparable", "latent",
]


def test_prompts_use_none_of_the_benchmark_vocabulary(cases):
    for case in cases:
        prompt = build_competence_prompt(case)
        text = (prompt.system + "\n" + prompt.user).lower()
        for phrase in BANNED:
            assert phrase not in text, f"{case.case_id} leaks '{phrase}'"


def test_reference_range_prompt_states_the_closed_convention(cases):
    for case in rr(cases):
        user = build_reference_range_prompt(case).user
        assert "CLOSED interval" in user
        assert "counts as within the range" in user


def test_reference_range_prompt_contains_the_numbers(cases):
    for case in rr(cases):
        user = build_reference_range_prompt(case).user
        assert case.analyte in user
        assert f"{case.result:g}" in user


def test_affine_prompt_states_the_equation_and_asks_for_a_number(cases):
    for case in af(cases):
        user = build_affine_prompt(case).user
        assert "y = a * x + b" in user
        assert f"a = {case.a:g}" in user and f"b = {case.b:g}" in user
        assert "plain JSON number" in user


def test_prompt_never_states_the_answer(cases):
    for case in af(cases):
        user = build_affine_prompt(case).user
        # the answer must not appear as a standalone token anywhere
        assert f"x = {case.expected_x:g}" not in user
    for case in rr(cases):
        user = build_reference_range_prompt(case).user
        assert f'"{case.expected_classification.value}"' not in user.split("Answer with")[0]


def test_prompts_are_deterministic(cases):
    for case in cases:
        assert build_competence_prompt(case) == build_competence_prompt(case)


def test_prompt_shell_is_identical_within_a_skill(cases):
    shells = {build_affine_prompt(c).user.split("For this problem")[0] for c in af(cases)}
    assert len(shells) == 1


# ============================================================== config guards
class TestCompetenceConfig:
    def test_thresholds_load_from_config(self, cfg):
        assert cfg.competence.overall_min == 0.90
        assert cfg.competence.per_skill_min == 0.85
        assert cfg.competence.cases_per_model == 30

    def test_overall_min_must_match_the_continuation_criterion(self, cfg):
        from src.generate_cases import PilotConfig

        broken = cfg.model_dump()
        broken["competence"]["overall_min"] = 0.95
        with pytest.raises(ValueError, match="must equal"):
            PilotConfig(**broken)

    def test_case_count_must_support_the_planned_split(self, cfg):
        from src.generate_cases import PilotConfig

        broken = cfg.model_dump()
        broken["competence"]["cases_per_model"] = 20  # 10 per skill, not divisible by 3
        with pytest.raises(ValueError, match="below/within/above"):
            PilotConfig(**broken)

    def test_odd_case_count_is_rejected(self, cfg):
        from src.generate_cases import PilotConfig

        broken = cfg.model_dump()
        broken["competence"]["cases_per_model"] = 31
        with pytest.raises(ValueError, match="must be even"):
            PilotConfig(**broken)

    def test_per_skill_floor_above_overall_floor_is_rejected(self, cfg):
        from src.generate_cases import PilotConfig

        broken = cfg.model_dump()
        broken["competence"]["per_skill_min"] = 0.95
        with pytest.raises(ValueError, match="could never bind"):
            PilotConfig(**broken)


# ================================================= the generator cannot cheat
def test_builder_raises_rather_than_mislabelling(cfg):
    """If the constructed number lands in the wrong class, the generator must
    fail loudly instead of relabelling the case to match what it produced."""
    with pytest.raises(RuntimeError, match="lands in"):
        # asking for BELOW while forcing the result onto the upper endpoint
        build_reference_range_case(0, RangeClass.BELOW, BoundaryKind.AT_UPPER, cfg.master_seed)


def test_affine_builder_is_a_pure_function_of_its_inputs(cfg):
    first = build_affine_case(15, 0, cfg.master_seed)
    second = build_affine_case(15, 0, cfg.master_seed)
    assert first.model_dump() == second.model_dump()
    assert math.isclose(first.a * first.expected_x + first.b, first.y, abs_tol=1e-6)
