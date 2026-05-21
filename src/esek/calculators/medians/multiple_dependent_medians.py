"""Multiple dependent-samples median description and robust ANOVA.

Implements the methods from the legacy
``stats/Calculator/Medians/Multi_Dep_Medians.py`` source file.  That file had
NO R/rpy2 dependencies — all computations were pure Python / NumPy / SciPy.

Classes
-------
MultipleDependentMedians
    Robust descriptive statistics and one-way trimmed-mean ANOVA for multiple
    within-subjects (dependent/repeated-measures) groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, trim_mean


@dataclass(frozen=True)
class MultipleDependentMediansResult:
    """Result for multiple dependent groups median analysis.

    Attributes
    ----------
    descriptives : pd.DataFrame
        Per-group median, trimmed mean, MAD, IQR.
    robust_anova : dict[str, Any]
        Statistics from the one-way robust ANOVA on trimmed means.
    metadata : dict[str, Any]
    """

    descriptives: pd.DataFrame
    robust_anova: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _winsorized_variance(x: np.ndarray, trimming: float = 0.2) -> float:
    """Winsorized variance following the Wilcox (2012) definition.

    Parameters
    ----------
    x:
        1-D numeric array.
    trimming:
        Symmetric trimming proportion (default 0.2 → 20 % each side).

    Returns
    -------
    float
        The winsorized sample variance.

    References
    ----------
    - Wilcox, R. R. (2012). *Introduction to Robust Estimation and
      Hypothesis Testing* (3rd ed.). Academic Press.
    """
    y = np.sort(x.astype(float))
    n = len(y)
    ibot = int(np.floor(trimming * n))
    itop = n - ibot
    xbot = y[ibot]
    xtop = y[itop - 1]
    y = np.clip(y, xbot, xtop)
    return float(np.var(y, ddof=1))


class MultipleDependentMedians:
    """Robust description and ANOVA for multiple within-subjects groups.

    No R/rpy2 dependencies are required.
    """

    @staticmethod
    def from_data(
        groups: dict[str, np.ndarray] | Sequence[np.ndarray],
        trimming: float = 0.2,
        confidence_level: float = 0.95,
    ) -> MultipleDependentMediansResult:
        """Compute descriptive statistics and robust ANOVA for multiple groups.

        Parameters
        ----------
        groups:
            Either a dict ``{group_name: array}`` or a sequence of arrays.
            All arrays must be equal length (balanced design assumed).
        trimming:
            Symmetric trimming proportion for the robust ANOVA (default 0.2).
        confidence_level:
            Desired confidence level (not used in descriptives, reserved for
            future CI computation).

        Returns
        -------
        MultipleDependentMediansResult
        """
        if isinstance(groups, dict):
            names = list(groups.keys())
            arrays = [np.asarray(v, dtype=float) for v in groups.values()]
        else:
            arrays = [np.asarray(g, dtype=float) for g in groups]
            names = [f"Group_{i + 1}" for i in range(len(arrays))]

        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("All groups must have the same number of observations.")
        if n < 3:
            raise ValueError("At least 3 observations per group are required.")
        if not (0.0 < trimming < 0.5):
            raise ValueError("trimming must be in (0, 0.5).")

        # ── Descriptive statistics ─────────────────────────────────────────
        rows: list[dict[str, Any]] = []
        for name, arr in zip(names, arrays):
            rows.append(
                {
                    "group": name,
                    "n": n,
                    "median": float(np.median(arr)),
                    "trimmed_mean": float(trim_mean(arr, proportiontocut=trimming)),
                    "mad": float(median_abs_deviation(arr)),
                    "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                    "mean": float(np.mean(arr)),
                    "sd": float(np.std(arr, ddof=1)),
                }
            )
        descriptives = pd.DataFrame(rows).set_index("group")

        # ── Robust one-way repeated-measures ANOVA (Wilcox, 2012) ─────────
        # Using trimmed means: g = floor(n * trimming), h = n - 2g
        g_trim = int(np.floor(n * trimming))
        h = n - 2 * g_trim
        j = len(arrays)  # number of groups

        trimmed_means = np.array(
            [float(trim_mean(a, proportiontocut=trimming)) for a in arrays]
        )
        grand_trim_mean = float(np.mean(trimmed_means))

        # Gc statistic (between-groups sum of squares on trimmed means)
        gc = float(h * np.sum((trimmed_means - grand_trim_mean) ** 2))

        # Winsorized variances per group
        win_vars = np.array([_winsorized_variance(a, trimming) for a in arrays])

        # Total winsorized variance (pooled)
        df_trim = (h - 1) * j

        robust_anova: dict[str, Any] = {
            "n_per_group": n,
            "n_groups": j,
            "trimming_level": trimming,
            "g_trimmed_per_side": g_trim,
            "h_effective": h,
            "gc_statistic": gc,
            "trimmed_means": {name: float(tm) for name, tm in zip(names, trimmed_means)},
            "grand_trimmed_mean": grand_trim_mean,
            "winsorized_variances": {name: float(wv) for name, wv in zip(names, win_vars)},
            "df": df_trim,
        }

        return MultipleDependentMediansResult(
            descriptives=descriptives,
            robust_anova=robust_anova,
            metadata={
                "confidence_level": confidence_level,
                "trimming": trimming,
            },
        )


__all__ = ["MultipleDependentMedians", "MultipleDependentMediansResult"]
