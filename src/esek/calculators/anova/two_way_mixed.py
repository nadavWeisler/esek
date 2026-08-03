"""Two-way mixed ANOVA (between × within).

Migrated from
``stats/Calculator/MultipleMeans/ANOVA/Two-Way-Mixed_Final_ANOVA.ipynb`` on the
``dev`` branch.  The notebook relied on R (``afex`` / ``emmeans`` via rpy2);
this port uses ``pingouin.mixed_anova`` and ``pingouin.pairwise_tests`` plus
existing η² CI helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg

from ...confidence_intervals.ci_eta_squared import EtaSquaredCI, EtaSquaredCIResult
from ...core import InvalidInputError
from ...core.validation import validate_confidence_level, validate_non_empty


@dataclass(frozen=True)
class MixedANOVAEffect:
    """One ANOVA effect row with optional partial-η² CI."""

    source: str
    ss: float
    df1: float
    df2: float
    ms: float
    f_statistic: float
    p_value: float
    partial_eta_squared: float
    eta_squared_ci: EtaSquaredCIResult | None = None


@dataclass(frozen=True)
class TwoWayMixedANOVAResult:
    """Result of a two-way mixed ANOVA."""

    between: str
    within: str
    subject: str
    dependent: str
    confidence_level: float
    effects: tuple[MixedANOVAEffect, ...]
    pairwise: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TwoWayMixedANOVA:
    """Between × within mixed ANOVA via pingouin."""

    @staticmethod
    def from_data(
        data: pd.DataFrame,
        dependent: str,
        subject: str,
        within: str,
        between: str,
        confidence_level: float = 0.95,
        include_pairwise: bool = True,
        padjust: str = "holm",
    ) -> TwoWayMixedANOVAResult:
        """Fit a two-way mixed ANOVA.

        Parameters
        ----------
        data:
            Long-format data frame.
        dependent:
            Dependent-variable column name.
        subject:
            Subject-id column name.
        within:
            Within-subjects factor column name.
        between:
            Between-subjects factor column name.
        confidence_level:
            Confidence level for partial-η² CIs.
        include_pairwise:
            If True, attach pingouin pairwise tests for both factors.
        padjust:
            Multiple-comparison adjustment for pairwise tests.
        """
        validate_confidence_level(confidence_level)
        if not isinstance(data, pd.DataFrame):
            raise InvalidInputError("'data' must be a pandas DataFrame.")
        validate_non_empty(data, name="data")
        for col in (dependent, subject, within, between):
            if col not in data.columns:
                raise InvalidInputError(f"Column {col!r} is missing from data.")

        anova = pg.mixed_anova(
            data=data,
            dv=dependent,
            within=within,
            subject=subject,
            between=between,
            correction="auto",
            effsize="np2",
        )

        effects: list[MixedANOVAEffect] = []
        for _, row in anova.iterrows():
            source = str(row["Source"])
            df1 = float(row["DF1"])
            df2 = float(row["DF2"])
            f_stat = float(row["F"])
            np2 = float(row["np2"])
            eta_ci = None
            if df1 > 0 and df2 > 0 and np.isfinite(f_stat) and f_stat >= 0:
                df1_i = max(int(round(df1)), 1)
                df2_i = max(int(round(df2)), 1)
                try:
                    eta_ci = EtaSquaredCI.from_f(
                        f_statistic=f_stat,
                        df1=df1_i,
                        df2=df2_i,
                        confidence_level=confidence_level,
                    )
                except Exception:
                    eta_ci = None
            p_key = "p-unc" if "p-unc" in row.index else "p_unc"
            effects.append(
                MixedANOVAEffect(
                    source=source,
                    ss=float(row["SS"]) if "SS" in row and pd.notna(row["SS"]) else float("nan"),
                    df1=df1,
                    df2=df2,
                    ms=float(row["MS"]) if "MS" in row and pd.notna(row["MS"]) else float("nan"),
                    f_statistic=f_stat,
                    p_value=float(row[p_key]),
                    partial_eta_squared=np2,
                    eta_squared_ci=eta_ci,
                )
            )

        pairwise = None
        if include_pairwise:
            frames = []
            pw_within = pg.pairwise_tests(
                data=data,
                dv=dependent,
                within=within,
                subject=subject,
                padjust=padjust,
            ).copy()
            pw_within.insert(0, "factor", within)
            frames.append(pw_within)
            pw_between = pg.pairwise_tests(
                data=data,
                dv=dependent,
                between=between,
                subject=subject,
                padjust=padjust,
            ).copy()
            pw_between.insert(0, "factor", between)
            frames.append(pw_between)
            pairwise = pd.concat(frames, ignore_index=True)

        return TwoWayMixedANOVAResult(
            between=between,
            within=within,
            subject=subject,
            dependent=dependent,
            confidence_level=float(confidence_level),
            effects=tuple(effects),
            pairwise=pairwise,
            metadata={"anova_table": anova},
        )
