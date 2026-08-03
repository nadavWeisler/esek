"""Intraclass correlation coefficients (ICC).

Migrated from ``stats/Calculator/MeasureAgreements/ICC.ipynb`` on the ``dev``
branch.  The notebook used ``pymer4``/lme4; this port uses
``pingouin.intraclass_corr`` for a pure-Python implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg

from ...core import InvalidInputError
from ...core.validation import validate_confidence_level, validate_non_empty

# pingouin >=0.5 uses Shrout–Fleiss labels like ICC(1,1); older docs used ICC1.
_ICC_TYPE_ALIASES = {
    "ICC1": ("ICC1", "ICC(1,1)"),
    "ICC2": ("ICC2", "ICC(A,1)"),
    "ICC3": ("ICC3", "ICC(C,1)"),
    "ICC1k": ("ICC1k", "ICC(1,k)"),
    "ICC2k": ("ICC2k", "ICC(A,k)"),
    "ICC3k": ("ICC3k", "ICC(C,k)"),
}


@dataclass(frozen=True)
class ICCTypeResult:
    """One ICC type with inferential statistics."""

    icc_type: str
    icc: float
    f_statistic: float
    df1: float
    df2: float
    p_value: float
    ci: tuple[float, float]


@dataclass(frozen=True)
class ICCResult:
    """Collection of Shrout–Fleiss ICC estimates."""

    confidence_level: float
    n_subjects: int
    n_raters: int
    icc1: ICCTypeResult
    icc2: ICCTypeResult
    icc3: ICCTypeResult
    icc1k: ICCTypeResult
    icc2k: ICCTypeResult
    icc3k: ICCTypeResult


class IntraclassCorrelation:
    """Intraclass correlation coefficients for subject × rater matrices."""

    @staticmethod
    def from_data(
        data: np.ndarray | Sequence[Sequence[float]],
        confidence_level: float = 0.95,
    ) -> ICCResult:
        """Compute ICC1/2/3 (single and average) from ratings.

        Parameters
        ----------
        data:
            Matrix with subjects in rows and raters in columns.
        confidence_level:
            Confidence level in ``(0, 1)``.  Note: pingouin currently returns
            95% CIs; the field is retained for API consistency.
        """
        validate_confidence_level(confidence_level)
        arr = np.asarray(data, dtype=float)
        validate_non_empty(arr, name="data")
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise InvalidInputError(
                "'data' must be a 2-D matrix with at least 2 subjects and 2 raters."
            )
        if np.any(~np.isfinite(arr)):
            raise InvalidInputError("'data' must contain only finite values.")

        n_subjects, n_raters = arr.shape
        long = (
            pd.DataFrame(arr, columns=[f"r{j}" for j in range(n_raters)])
            .reset_index(names="subject")
            .melt(id_vars="subject", var_name="rater", value_name="rating")
        )
        table = pg.intraclass_corr(
            data=long,
            targets="subject",
            raters="rater",
            ratings="rating",
        )
        by_type = {str(row["Type"]): row for _, row in table.iterrows()}

        def _one(canonical: str) -> ICCTypeResult:
            row = None
            matched = None
            for alias in _ICC_TYPE_ALIASES[canonical]:
                if alias in by_type:
                    row = by_type[alias]
                    matched = alias
                    break
            if row is None:
                raise InvalidInputError(
                    f"pingouin did not return ICC type for {canonical!r} "
                    f"(available: {sorted(by_type)})."
                )
            ci_key = "CI95%" if "CI95%" in row.index else "CI95"
            ci = row[ci_key]
            ci_tuple = (float(ci[0]), float(ci[1]))
            p_key = "pval" if "pval" in row.index else "p"
            return ICCTypeResult(
                icc_type=str(matched),
                icc=float(row["ICC"]),
                f_statistic=float(row["F"]),
                df1=float(row["df1"]),
                df2=float(row["df2"]),
                p_value=float(row[p_key]),
                ci=ci_tuple,
            )

        return ICCResult(
            confidence_level=float(confidence_level),
            n_subjects=n_subjects,
            n_raters=n_raters,
            icc1=_one("ICC1"),
            icc2=_one("ICC2"),
            icc3=_one("ICC3"),
            icc1k=_one("ICC1k"),
            icc2k=_one("ICC2k"),
            icc3k=_one("ICC3k"),
        )
