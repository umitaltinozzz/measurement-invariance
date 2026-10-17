"""Small-sample confidence intervals and exact paired-test helpers for reports."""

from __future__ import annotations

from math import comb, sqrt
from typing import Optional, Tuple


def wilson_interval(successes: int, total: int) -> Optional[Tuple[float, float]]:
    """Return a two-sided 95% Wilson score interval, or None for an empty slice."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return None

    z = 1.959963984540054
    p = successes / total
    z2 = z * z
    scale = 1 + z2 / total
    center = (p + z2 / (2 * total)) / scale
    half_width = z * sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / scale
    return max(0.0, center - half_width), min(1.0, center + half_width)


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts."""
    if first_only < 0 or second_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))
