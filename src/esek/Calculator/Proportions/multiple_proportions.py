# src/esek/Calculator/MultipleProportions/multiple_proportions.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import cochrans_q, StratifiedTable

from ...utils import res, interfaces


@dataclass
class CochranQResults:
    n: int
    k: int  # number of measures (columns)
    q_stat: float
    df: int
    p_value: float
    variance_q: float
    mean_q: float
    r_effect_size: float  # chance-corrected Q-based R (Berry et al., 2010)


@dataclass
class GoodnessOfFitResults:
    levels: Sequence[str]
    observed: np.ndarray  # counts
    expected: np.ndarray  # counts
    n: int
    df: int
    chi_square: float
    p_value_chi_square: float
    wilks_g: float
    cohens_w: float
    max_corrected_lambda: float  # Chi-square / max Chi-square
    max_corrected_gamma: float  # G^2 / max G^2
    chance_corrected_R: float
    variance_R: float
    mean_R: float


class MultipleProportions(interfaces.AbstractTest):
    """
    - Cochran's Q (wide, binary 0/1 data per subject)
    - One-way categorical goodness-of-fit (from a 1D vector of labels + optional expectations)
    """

    # ---------- Cochran's Q ----------
    @staticmethod
    def cochran_q_from_wide_binary(
        df_like: pd.DataFrame | np.ndarray,
    ) -> CochranQResults:
        """
        df_like: shape (n_subjects, k_measures) with 0/1 values.
        """
        X = pd.DataFrame(df_like).astype(float)
        n, k = X.shape
        df = k - 1

        # row sums and per-row success probability estimates (Pi_i)
        row_sums = X.sum(axis=1).to_numpy()
        Pis = row_sums / k

        # Cochran's Q (statsmodels)
        q_test = cochrans_q(X)
        q_stat = float(q_test.statistic)
        pval = float(q_test.pvalue)

        # Berry et al. (2010) chance-corrected Q-based effect size R
        # Using your formulas verbatim:
        A = float(Pis.sum())
        B = n - A
        C = df / (2.0 * float((Pis * (1.0 - Pis)).sum()) if n > 0 else 1.0)

        variance_q = (q_stat / C - 2.0 * A * B) / (n * (n - 1)) if n > 1 else 0.0
        mean_q = (
            (2.0 / (n * (n - 1))) * (A * B - float((Pis * (1.0 - Pis)).sum()))
            if n > 1
            else 0.0
        )
        r_es = 1.0 - (variance_q / mean_q) if mean_q != 0 else 0.0

        return CochranQResults(
            n=n,
            k=k,
            q_stat=q_stat,
            df=df,
            p_value=pval,
            variance_q=float(variance_q),
            mean_q=float(mean_q),
            r_effect_size=float(r_es),
        )

    # ---------- Goodness of Fit ----------
    @staticmethod
    def goodness_of_fit_from_frequency(
        values: Iterable,  # 1D labels (any hashable)
        expected_proportions: Optional[Sequence[float]] = None,
        expected_frequencies: Optional[Sequence[float]] = None,
        expected_ratios: Optional[Sequence[float]] = None,
    ) -> GoodnessOfFitResults:

        s = pd.Series(list(values))
        observed_counts = s.value_counts(sort=False)  # keep natural label order
        levels = observed_counts.index.astype(str).tolist()
        observed = observed_counts.to_numpy(dtype=float)
        n = int(observed.sum())
        k = observed.size
        df = k - 1

        if expected_proportions is not None:
            exp = np.asarray(expected_proportions, dtype=float)
            exp = (exp / exp.sum()) * n
        elif expected_frequencies is not None:
            exp = np.asarray(expected_frequencies, dtype=float)
        elif expected_ratios is not None:
            r = np.asarray(expected_ratios, dtype=float)
            exp = (r / r.sum()) * n
        else:
            exp = np.full(k, n / k, dtype=float)

        # align sizes defensively
        if exp.shape[0] != k:
            raise ValueError(
                "Expected vector length must equal number of observed levels."
            )

        # Chi-square
        with np.errstate(divide="ignore", invalid="ignore"):
            chi_sq = float(np.nansum((observed - exp) ** 2 / exp))
        p_chi = float(chi2.sf(abs(chi_sq), df))

        # Wilks' G^2
        obs_p = observed / n
        exp_p = exp / n
        # guard zeros in log
        mask = (obs_p > 0) & (exp_p > 0)
        wilks_g = float(2 * n * np.sum(obs_p[mask] * np.log(obs_p[mask] / exp_p[mask])))

        # Cohen's w
        cohens_w = float(np.sqrt(chi_sq / n)) if n > 0 else 0.0

        # maximum-corrected versions
        q_chi = float(exp.min())
        max_chi = (n * (n - q_chi)) / q_chi if q_chi > 0 else np.inf
        max_lambda = (
            float(chi_sq / max_chi) if np.isfinite(max_chi) and max_chi > 0 else 0.0
        )

        q_wilks = float(exp_p.min())
        max_g = -2 * n * np.log(q_wilks) if q_wilks > 0 else np.inf
        max_gamma = float(wilks_g / max_g) if np.isfinite(max_g) and max_g > 0 else 0.0

        # chance-corrected R (your formulas)
        variance_R = float((1.0 / k) * np.sum((obs_p - exp_p) ** 2))
        # pairwise mean of squared diffs between all obs_p and all exp_p
        mean_R = float(
            (1.0 / (k**2)) * np.sum([(oi - ej) ** 2 for oi in obs_p for ej in exp_p])
        )
        cc_R = float(variance_R / mean_R) if mean_R != 0 else 0.0

        return GoodnessOfFitResults(
            levels=levels,
            observed=observed.astype(float),
            expected=exp.astype(float),
            n=n,
            df=df,
            chi_square=chi_sq,
            p_value_chi_square=p_chi,
            wilks_g=wilks_g,
            cohens_w=cohens_w,
            max_corrected_lambda=max_lambda,
            max_corrected_gamma=max_gamma,
            chance_corrected_R=cc_R,
            variance_R=variance_R,
            mean_R=mean_R,
        )
