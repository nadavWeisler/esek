"""Descriptive and inferential statistics containers.

These are plain (non-frozen) dataclasses used internally by calculators
to carry intermediate results before building the final frozen result
objects. They preserve backward compatibility with the existing
``utils/results.py`` classes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """A simple (lower, upper) confidence interval container."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        self.lower = float(self.lower)
        self.upper = float(self.upper)

    @property
    def ci(self) -> tuple[float, float]:
        return (self.lower, self.upper)


@dataclass
class DescriptiveStatistics:
    """Mean and standard deviation for a sample or population."""

    mean: float
    sd: float


@dataclass
class SampleStatistics(DescriptiveStatistics):
    """Descriptive statistics for one sample in a test."""

    size: int = 0
    diff_mean: float | None = None
    diff_sd: float | None = None
    population_sd_diff: float | None = None
    population_mean: float | None = None


@dataclass
class GroupStatistics(DescriptiveStatistics):
    """Descriptive statistics for one group in an independent-samples test."""

    median: float | None = None
    median_absolute_deviation: float | None = None
    diff_median: float | None = None
    sample_size: int | None = None
    u_statistic: float | None = None
    w_statistic: float | None = None
    mean_rank: float | None = None
    population_sd: float | None = None
    mean_diff: float | None = None
    sd_diff: float | None = None


@dataclass
class InferentialStatistics:
    """Test statistic and p-value from an inferential test."""

    p_value: float
    score: float
    standard_error: float | None = None
    degrees_of_freedom: float | None = None
    means_difference: float | None = None


@dataclass
class ApproximatedStandardError:
    """Multiple SE approximations for an effect size (e.g. Cohen's d).

    Attributes correspond to the formulas by Morris, Hedges, Hedges-Olkin,
    MLE, large-N, and Hunter & Schmidt.
    """

    true_se: float
    morris: float
    hedges: float
    hedges_olkin: float
    mle: float
    large_n: float
    hunter_and_schmidt: float
