"""Module for Ordinal by Interval correlation measures."""

import math
from collections import Counter

import numpy as np
from scipy.stats import (
    chi2,
    median_abs_deviation,
    norm,
    rankdata,
    spearmanr,
    t,
)
from sklearn.covariance import MinCovDet


def _ideal_fourth_iqr(x):
    """Compute the ideal fourth inter-quartile range (Wilcox, 2012)."""
    n = len(x)
    j = int(np.floor(n / 4 + 5 / 12))
    y = np.sort(x)
    g = (n / 4) - j + (5 / 12)
    low = (1 - g) * y[j - 1] + g * y[j]
    k = n - j + 1
    up = (1 - g) * y[k - 1] + g * y[k - 2]
    return up - low


def _skipped_correlation(x, y, confidence_level):
    """Compute skipped Spearman correlation (outlier-robust).

    Uses the minimum covariance determinant (MCD) to flag bivariate outliers
    via both IQR-based and MAD-based thresholds (Wilcox, 2012).

    Parameters
    ----------
    x, y : array-like
        Data vectors of the same length.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack((x, y))
    nrows = X.shape[0]
    gval = np.sqrt(chi2.ppf(0.975, 2))

    try:
        center = MinCovDet(random_state=42).fit(X).location_
    except Exception:
        rho, p_val = spearmanr(x, y)
        rho = float(np.clip(rho, -0.9999999, 0.9999999))
        results = {
            "Skipped Correlation IQR based": rho,
            "Skipped Correlation MAD based": rho,
            "Skipped Correlation IQR based p-value": p_val,
            "Skipped Correlation MAD based p-value": p_val,
            "Skipped Correlation IQR based CI's": [float("nan"), float("nan")],
            "Skipped Correlation MAD based CI's": [float("nan"), float("nan")],
        }
        return "\n".join([f"{k}: {v}" for k, v in results.items()])

    B = X - center
    bot = (B ** 2).sum(axis=1)
    dis = np.zeros((nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:
            dis[i, :] = np.linalg.norm(B.dot(B[i, :, None]) * B[i, :] / bot[i], axis=1)

    iqr = np.apply_along_axis(_ideal_fourth_iqr, 1, dis)
    thresh_iqr = np.median(dis, axis=1) + gval * iqr
    outliers_iqr = np.apply_along_axis(np.greater, 0, dis, thresh_iqr).any(axis=0)

    mad = np.apply_along_axis(median_abs_deviation, 1, dis)
    thresh_mad = np.median(dis, axis=1) + gval * mad
    outliers_mad = np.apply_along_axis(np.greater, 0, dis, thresh_mad).any(axis=0)

    rho_iqr, p_iqr = spearmanr(X[~outliers_iqr, 0], X[~outliers_iqr, 1])
    rho_mad, p_mad = spearmanr(X[~outliers_mad, 0], X[~outliers_mad, 1])
    rho_iqr = float(np.clip(rho_iqr, -0.9999999, 0.9999999))
    rho_mad = float(np.clip(rho_mad, -0.9999999, 0.9999999))

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    n_iqr = len(X[~outliers_iqr, 0])
    n_mad = len(X[~outliers_mad, 0])

    zr_iqr = 0.5 * np.log((1 + rho_iqr ** 2) / (1 - rho_iqr))
    se_iqr = np.sqrt((1 + rho_iqr ** 2 / 2) / (n_iqr - 3))
    lower_iqr = (math.exp(2 * (zr_iqr - zcrit * se_iqr)) - 1) / (math.exp(2 * (zr_iqr - zcrit * se_iqr)) + 1)
    upper_iqr = (math.exp(2 * (zr_iqr + zcrit * se_iqr)) - 1) / (math.exp(2 * (zr_iqr + zcrit * se_iqr)) + 1)

    zr_mad = 0.5 * np.log((1 + rho_mad ** 2) / (1 - rho_mad))
    se_mad = np.sqrt((1 + rho_mad ** 2 / 2) / (n_mad - 3))
    lower_mad = (math.exp(2 * (zr_mad - zcrit * se_mad)) - 1) / (math.exp(2 * (zr_mad - zcrit * se_mad)) + 1)
    upper_mad = (math.exp(2 * (zr_mad + zcrit * se_mad)) - 1) / (math.exp(2 * (zr_mad + zcrit * se_mad)) + 1)

    results = {
        "Skipped Correlation IQR based": rho_iqr,
        "Skipped Correlation MAD based": rho_mad,
        "Skipped Correlation IQR based p-value": p_iqr,
        "Skipped Correlation MAD based p-value": p_mad,
        "Skipped Correlation IQR based CI's": [lower_iqr, upper_iqr],
        "Skipped Correlation MAD based CI's": [lower_mad, upper_mad],
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _spearman_correlation(x, y, confidence_level):
    """Compute Spearman correlation with multiple SE/CI variants.

    Parameters
    ----------
    x, y : array-like
        Data vectors.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = rankdata(x)
    ry = rankdata(y)
    n = len(x)

    rho, p_rho = spearmanr(x, y)
    rho_val = float(np.clip(rho if np.isscalar(rho) else float(rho), -0.9999999, 0.9999999))
    zrho = math.atanh(rho_val)

    se_fieller = np.sqrt(1.06 / (n - 3))
    se_cc = np.sqrt(1 / (n - 2)) + (abs(zrho) / (6 * n + 4 * np.sqrt(n)))
    se_bw = (1 + rho_val / 2) / (n - 3)

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    tcrit = t.ppf(1 - (1 - confidence_level) / 2, n - 2)

    lower_bw = math.tanh(zrho - zcrit * se_bw)
    upper_bw = math.tanh(zrho + zcrit * se_bw)
    lower_fieller = math.tanh(zrho - zcrit * se_fieller)
    upper_fieller = math.tanh(zrho + zcrit * se_fieller)
    lower_cc = math.tanh(zrho - zcrit * se_cc)
    upper_cc = math.tanh(zrho + zcrit * se_cc)
    lower_woods = math.tanh(zrho - tcrit * np.sqrt(1 / (n - 3)))
    upper_woods = math.tanh(zrho + tcrit * np.sqrt(1 / (n - 3)))
    lower_fisher = math.tanh(zrho - zcrit * np.sqrt(1 / (n - 3)))
    upper_fisher = math.tanh(zrho + zcrit * np.sqrt(1 / (n - 3)))

    # Ties-corrected Spearman (Oyeka & Nwankwo Chike, 2014)
    exp_rx = x.argsort().argsort() + 1
    exp_ry = y.argsort().argsort() + 1
    diff_rx = exp_rx - rx
    diff_ry = exp_ry - ry
    prod_x = exp_rx * diff_rx
    prod_y = exp_ry * diff_ry
    di_x = diff_rx ** 2 / 2
    di_y = diff_ry ** 2 / 2
    pi_x = sum(1 for d in diff_rx if d != 0) / n
    pi_y = sum(1 for d in diff_ry if d != 0) / n
    multi_ranked = rx * ry
    term_x = (n * (n ** 2 - 1)) / 12 - 2 * pi_x * (sum(prod_x) - sum(di_x))
    term_y = (n * (n ** 2 - 1)) / 12 - 2 * pi_y * (sum(prod_y) - sum(di_y))
    numerator = np.sum(multi_ranked) - (n * (n + 1) ** 2) / 4
    denominator = np.sqrt(term_x * term_y)
    rho_oyeka = float(np.clip(numerator / denominator, -0.9999999, 0.9999999))
    p_oyeka = t.sf(abs(rho_oyeka * ((n - 2) / (1 - rho_oyeka ** 2))), n - 2)
    se_bw_oyeka = (1 + rho_oyeka / 2) / (n - 3)
    lower_oyeka = math.tanh(math.atanh(rho_oyeka) - zcrit * se_bw_oyeka)
    upper_oyeka = math.tanh(math.atanh(rho_oyeka) + zcrit * se_bw_oyeka)

    # Ties-corrected Spearman (Taylor, 1964)
    d_sq = np.sum((rx - ry) ** 2)
    freq_x = np.array([c for c in Counter(x).values() if c > 1])
    freq_y = np.array([c for c in Counter(y).values() if c > 1])
    tx = sum(freq_x ** 3 - freq_x) / 12
    ty = sum(freq_y ** 3 - freq_y) / 12
    rho_taylor = float(np.clip(1 - (6 * (d_sq + tx + ty)) / (n * (n ** 2 - 1)), -0.9999999, 0.9999999))
    p_taylor = t.sf(abs(rho_oyeka * ((n - 2) / (1 - rho_taylor ** 2))), n - 2)
    se_bw_taylor = (1 + rho_taylor / 2) / (n - 3)
    lower_taylor = math.tanh(math.atanh(rho_taylor) - zcrit * se_bw_taylor)
    upper_taylor = math.tanh(math.atanh(rho_taylor) + zcrit * se_bw_taylor)

    fmt_p = "{:.3f}".format(p_rho).lstrip("0") if p_rho >= 0.001 else "\033[3mp\033[0m < .001"
    fmt_p_oyeka = "{:.3f}".format(p_oyeka).lstrip("0") if p_oyeka >= 0.001 else "\033[3mp\033[0m < .001"
    fmt_p_taylor = "{:.3f}".format(p_taylor).lstrip("0") if p_taylor >= 0.001 else "\033[3mp\033[0m < .001"

    results = {
        "Spearman": rho_val,
        "Spearman p-value": p_rho,
        "Standard Error (Bonett & Wright)": se_bw,
        "Standard Error (Fieller)": se_fieller,
        "Standard Error (Caruso and Cliff)": se_cc,
        "Confidence Intervals (Bonett Wright)": f"({round(lower_bw, 4)}, {round(upper_bw, 4)})",
        "Confidence Intervals (Fieller)": f"({round(lower_fieller, 4)}, {round(upper_fieller, 4)})",
        "Confidence Intervals (Caruso and Cliff)": f"({round(lower_cc, 4)}, {round(upper_cc, 4)})",
        "Confidence Intervals (Woods)": f"({round(lower_woods, 4)}, {round(upper_woods, 4)})",
        "Confidence Intervals (Fisher)": f"({round(lower_fisher, 4)}, {round(upper_fisher, 4)})",
        "Ties Corrected Spearman (Oyeka and Nwankwo Chike, 2014)": rho_oyeka,
        "Ties Corrected Spearman p-value (Oyeka and Nwankwo Chike, 2014)": p_oyeka,
        "Ties Corrected Spearman Standard Error (Oyeka and Nwankwo Chike, 2014)": se_bw_oyeka,
        "Ties Corrected Spearman CI's (Oyeka and Nwankwo Chike, 2014)": f"({round(lower_oyeka, 4)}, {round(upper_oyeka, 4)})",
        "Ties Corrected Spearman (Taylor, 1964)": rho_taylor,
        "Ties Corrected Spearman p-value (Taylor, 1964)": p_taylor,
        "Ties Corrected Spearman Standard Error (Taylor, 1964)": se_bw_taylor,
        "Ties Corrected Spearman CI's (Taylor, 1964)": f"({round(lower_taylor, 4)}, {round(upper_taylor, 4)})",
        "Statistical Line Spearman": (
            " \033[3mr\033[0m = {}, {}{}, {}% CI [{}, {}]".format(
                rho_val,
                "\033[3mp = \033[0m" if p_rho >= 0.001 else "",
                fmt_p,
                confidence_level,
                round(lower_bw, 3),
                round(upper_bw, 3),
            )
        ),
        "Statistical Line Corrected Spearman (Oyeka et al.)": (
            " {}{}, {}% CI [{}, {}]".format(
                rho_oyeka,
                "\033[3mp = \033[0m" if p_oyeka >= 0.001 else "",
                fmt_p_oyeka,
                confidence_level,
                round(lower_oyeka, 3),
                round(upper_oyeka, 3),
            )
        ),
        "Statistical Line (Taylor)": (
            " \033[3mr\033[0m = {}, {}{}, {}% CI [{}, {}]".format(
                rho_taylor,
                "\033[3mp = \033[0m" if p_taylor >= 0.001 else "",
                fmt_p_taylor,
                confidence_level,
                round(lower_taylor, 3),
                round(upper_taylor, 3),
            )
        ),
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _gaussian_rank_correlation(x, y, confidence_level=0.95):
    """Compute Gaussian rank correlation.

    Parameters
    ----------
    x, y : array-like
        Data vectors.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    from scipy.stats import norm as _norm

    nx = _norm.ppf(rankdata(x) / (len(x) + 1))
    ny = _norm.ppf(rankdata(y) / (len(y) + 1))
    rho_grc, p_grc = spearmanr(nx, ny)
    rho_grc = np.clip(rho_grc, -0.9999999, 0.9999999)
    n = len(x)
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    se_bw = (1 + rho_grc / 2) / (n - 3)
    lower = math.tanh(math.atanh(rho_grc) - zcrit * se_bw)
    upper = math.tanh(math.atanh(rho_grc) + zcrit * se_bw)

    results = {
        "Gaussian Rank Correlation": rho_grc,
        "Gaussian Rank Correlation p-value": p_grc,
        "Gaussian Rank Correlation CI's": [lower, upper],
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _bsmahal(a, b, n_boot=200):
    """Bootstrap Mahalanobis distances (Shepherd's Pi helper)."""
    n, _ = b.shape
    md = np.zeros((n, n_boot))
    nr = np.arange(n)
    xb = np.random.choice(nr, size=(n_boot, n), replace=True)
    for i in np.arange(n_boot):
        s1 = b[xb[i, :], 0]
        s2 = b[xb[i, :], 1]
        X = np.column_stack((s1, s2))
        mu = X.mean(0)
        _, R = np.linalg.qr(X - mu)
        sol = np.linalg.solve(R.T, (a - mu).T)
        md[:, i] = np.sum(sol ** 2, 0) * (n - 1)
    return md.mean(1)


def _shepherd(x, y, n_boot=200, confidence_level=0.95):
    """Compute Shepherd's Pi (outlier-robust Spearman).

    Parameters
    ----------
    x, y : array-like
        Data vectors.
    n_boot : int
        Number of bootstrap samples for outlier detection.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack((x, y))
    m = _bsmahal(X, X, n_boot)
    outliers = m >= 6
    rho_s, p_s = spearmanr(x[~outliers], y[~outliers])
    rho_s = float(np.clip(rho_s, -0.9999999, 0.9999999))
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    se_bw = (1 + rho_s / 2) / (n - 3)
    lower = math.tanh(math.atanh(rho_s) - zcrit * se_bw)
    upper = math.tanh(math.atanh(rho_s) + zcrit * se_bw)

    results = {
        "Shepherd's Pi": rho_s,
        "Shepherd's Pi p-value": p_s,
        "Shepherd's Pi CI's": [lower, upper],
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _ginis_gamma(x, y, confidence_level=0.95):
    """Compute Gini's Gamma association measure.

    Parameters
    ----------
    x, y : array-like
        Data vectors.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = rankdata(x)
    ry = rankdata(y)
    n = len(x)
    term1 = np.sum(np.abs((n + 1 - rx) - ry) - np.abs(rx - ry))
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

    if n % 2 == 0:
        gamma = term1 / (n ** 2 / 2)
        ase = np.sqrt((2 * (n ** 2 + 2)) / (3 * (n - 1) * n ** 2))
    else:
        gamma = term1 / ((n ** 2 - 1) / 2)
        ase = np.sqrt((2 * (n ** 2 + 3)) / (3 * (n - 1) * (n ** 2 - 1)))

    z = gamma / ase
    p_value = 2 * (1 - norm.cdf(np.abs(z)))
    lower = gamma - ase * zcrit
    upper = gamma + ase * zcrit

    results = {
        "Ginni's Gamma": gamma,
        "Ginni's Gamma p-value": p_value,
        "Ginni's Gamma Standard Error": ase,
        "Ginni's Gamma CI's": [lower, upper],
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


class OrdinalByInterval:
    """Ordinal by Interval correlation measures.

    Computes skipped correlation, Gaussian rank correlation, Gini's Gamma,
    Shepherd's Pi, and Spearman correlation from raw data or a contingency
    table.
    """

    @staticmethod
    def from_contingency_table(params: dict) -> dict:
        """Calculate measures from a contingency table.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Contingency Table"`` : 2-D numpy array.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).
            - ``"Number Of Bootstraps Samples"`` : int.

        Returns
        -------
        dict
            Result dictionary containing all correlation measures.
        """
        table = np.array(params["Contingency Table"])
        cl_pct = float(params["Confidence Level"])
        n_boot = int(params["Number Of Bootstraps Samples"])
        cl = cl_pct / 100

        var1 = np.array([
            j + 1
            for i in range(table.shape[0])
            for j in range(table.shape[1])
            for _ in range(table[i, j])
        ])
        var2 = np.array([
            i + 1
            for i in range(table.shape[0])
            for j in range(table.shape[1])
            for _ in range(table[i, j])
        ])

        return {
            "Contingency Table": table,
            "Skipped Correlation": _skipped_correlation(var1, var2, cl),
            "Gaussian Rank Correlation": _gaussian_rank_correlation(var1, var2, cl),
            "Ginni's Gamma": _ginis_gamma(var1, var2, cl),
            "Shepherd's Pi": _shepherd(var1, var2, n_boot, cl),
            "Spearman Correlation": _spearman_correlation(var1, var2, cl),
        }

    @staticmethod
    def from_data(params: dict) -> dict:
        """Calculate measures from raw data vectors.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Variable 1"`` : array-like, ordinal variable.
            - ``"Variable 2"`` : array-like, interval variable.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).
            - ``"Number Of Bootstraps Samples"`` : int.

        Returns
        -------
        dict
            Result dictionary containing all correlation measures.
        """
        var1 = np.asarray(params["Variable 1"], dtype=float)
        var2 = np.asarray(params["Variable 2"], dtype=float)
        cl_pct = float(params["Confidence Level"])
        n_boot = int(params["Number Of Bootstraps Samples"])
        cl = cl_pct / 100

        return {
            "Skipped Correlation": _skipped_correlation(var1, var2, cl),
            "Gaussian Rank Correlation": _gaussian_rank_correlation(var1, var2, cl),
            "Ginni's Gamma": _ginis_gamma(var1, var2, cl),
            "Shepherd's Pi": _shepherd(var1, var2, n_boot, cl),
            "Spearman Correlation": _spearman_correlation(var1, var2, cl),
        }
