"""Module for Ordinal Partial Correlation (Spearman-based)."""

import numpy as np
import pingouin as pg
from scipy.stats import norm


class OrdinalPartialCorrelation:
    """Spearman-based partial and semi-partial correlation.

    Uses pingouin's ``partial_corr`` with ``method="spearman"`` to compute
    partial and semi-partial correlations between an independent and a
    dependent variable while controlling for one or more covariates.

    The input DataFrame must contain columns named ``"independent_variable"``
    and ``"dependent_variable"``; all remaining columns are treated as
    covariates.
    """

    @staticmethod
    def from_data(params: dict) -> dict:
        """Calculate Spearman partial and semi-partial correlations.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Data"`` : pandas DataFrame with columns
              ``"independent_variable"``, ``"dependent_variable"``,
              and any number of covariate columns.
            - ``"Confidence Level"`` : float, confidence level as a
              percentage (e.g. 95 for 95%).

        Returns
        -------
        dict
            Dictionary containing sample size, partial correlation, semi-
            partial correlation, confidence intervals, p-values, and
            formatted statistical lines.
        """
        data = params["Data"]
        cl_pct = float(params["Confidence Level"])
        cl = cl_pct / 100

        covar_cols = [
            c for c in data.columns
            if c not in ("independent_variable", "dependent_variable")
        ]

        # Partial correlation
        pc_out = pg.partial_corr(
            data,
            x="independent_variable",
            y="dependent_variable",
            covar=covar_cols,
            method="spearman",
        )
        n = int(pc_out.values[0, 0])
        r_partial = float(pc_out.values[0, 1])
        p_partial = float(pc_out.values[0, 3])

        zcrit = norm.ppf(1 - (1 - cl) / 2)
        se_partial = (1 + r_partial / 2) / (n - 3)
        lower_partial = np.tanh(np.arctanh(r_partial) - zcrit * se_partial)
        upper_partial = np.tanh(np.arctanh(r_partial) + zcrit * se_partial)

        # Semi-partial correlation (only x is partialled)
        spc_out = pg.partial_corr(
            data,
            x="independent_variable",
            y="dependent_variable",
            x_covar=covar_cols,
            method="spearman",
        )
        r_semi = float(spc_out.values[0, 1])
        p_semi = float(spc_out.values[0, 3])
        se_semi = (1 + r_semi / 2) / (n - 3)
        lower_semi = np.tanh(np.arctanh(r_semi) - zcrit * se_semi)
        upper_semi = np.tanh(np.arctanh(r_semi) + zcrit * se_semi)

        fmt_p_partial = (
            "{:.3f}".format(p_partial).lstrip("0")
            if p_partial >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        fmt_p_semi = (
            "{:.3f}".format(p_semi).lstrip("0")
            if p_semi >= 0.001
            else "\033[3mp\033[0m < .001"
        )

        def _fmt_num(v):
            return ("-" if v < 0 else "") + str(np.abs(np.round(v, 3))).lstrip("0").rstrip("")

        def _fmt_cl(cl_val):
            pct = cl_val * 100
            return int(pct) if float(pct).is_integer() else "{:.1f}".format(pct).rstrip("0").rstrip(".")

        results = {
            "Sample Size": n,
            "Partial Correlation": r_partial,
            "Confidence Intervals Partial Correlation": [lower_partial, upper_partial],
            "p-value partial correlation": p_partial,
            "Semi Partial Correlation": r_semi,
            "Confidence Intervals Semi Partial Correlation": [lower_semi, upper_semi],
            "p-value Semi partial correlation": p_semi,
            "Statistical Line Partial Correlation": (
                "\033[3mr\033[0m({}) = {}, {}{}, {}% CI [{}, {}]".format(
                    n - 2,
                    _fmt_num(r_partial),
                    "\033[3mp = \033[0m" if p_partial >= 0.001 else "",
                    fmt_p_partial,
                    _fmt_cl(cl),
                    _fmt_num(lower_partial),
                    _fmt_num(upper_partial),
                )
            ),
            "Statistical Line Semi Partial Correlation": (
                "\033[3mr\033[0m({}) = {}, {}{}, {}% CI [{}, {}]".format(
                    n - 2,
                    _fmt_num(r_semi),
                    "\033[3mp = \033[0m" if p_semi >= 0.001 else "",
                    fmt_p_semi,
                    _fmt_cl(cl),
                    _fmt_num(lower_semi),
                    _fmt_num(upper_semi),
                )
            ),
        }
        return results
