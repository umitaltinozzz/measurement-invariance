import pytest

from src.reporting_stats import exact_mcnemar_p, wilson_interval


def test_wilson_interval_handles_small_samples_and_zero_denominator():
    lower, upper = wilson_interval(20, 25)
    assert lower == pytest.approx(0.609, abs=0.002)
    assert upper == pytest.approx(0.911, abs=0.002)
    assert wilson_interval(0, 0) is None


def test_exact_mcnemar_uses_two_sided_binomial_tail():
    assert exact_mcnemar_p(9, 2) == pytest.approx(0.0654296875)
    assert exact_mcnemar_p(7, 0) == pytest.approx(0.015625)
    assert exact_mcnemar_p(0, 0) == 1.0


@pytest.mark.parametrize("successes,total", [(-1, 3), (4, 3), (0, -1)])
def test_wilson_rejects_invalid_counts(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)
