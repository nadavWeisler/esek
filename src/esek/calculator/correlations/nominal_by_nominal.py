"""Module for Nominal by Nominal correlation measures."""

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, ncx2, norm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ncp_ci(chival, df, conf):
    """Compute non-central (pivotal) CI for a chi-square statistic.

    Parameters
    ----------
    chival : float
        Observed chi-square statistic.
    df : int
        Degrees of freedom.
    conf : float
        Confidence level (0–1).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the non-central CI.
    """
    if chival <= 0:
        return 0.0, 0.0

    def _low(chival, df, conf):
        bounds = [0.001, chival / 2, chival]
        ulim = 1 - (1 - conf) / 2
        if ncx2.cdf(chival, df, bounds[0]) < ulim:
            return [0, ncx2.cdf(chival, df, bounds[0])]
        while abs(ncx2.cdf(chival, df, bounds[1]) - ulim) > 0.00001:
            if ncx2.cdf(chival, df, bounds[1]) < ulim:
                bounds = [bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1]]
            else:
                bounds = [bounds[1], (bounds[1] + bounds[2]) / 2, bounds[2]]
        return [bounds[1]]

    def _high(chival, df, conf):
        uc = [chival, 2 * chival, 3 * chival]
        llim = (1 - conf) / 2
        while ncx2.cdf(chival, df, uc[0]) < llim:
            uc = [uc[0] / 4, uc[0], uc[2]]
        while ncx2.cdf(chival, df, uc[2]) > llim:
            uc = [uc[0], uc[2], uc[2] + chival]
        diff = 1
        while diff > 0.00001:
            if ncx2.cdf(chival, df, uc[1]) < llim:
                uc = [uc[0], (uc[0] + uc[1]) / 2, uc[1]]
            else:
                uc = [uc[1], (uc[1] + uc[2]) / 2, uc[2]]
            diff = abs(ncx2.cdf(chival, df, uc[1]) - llim)
        return uc[1]

    return _low(chival, df, conf)[0], _high(chival, df, conf)


def _berry_mielke_max_corrected_matrix(matrix):
    """Build the Berry-Mielke maximum-corrected contingency matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Observed contingency table.

    Returns
    -------
    numpy.ndarray
        Maximum-corrected matrix.
    """
    r, c = matrix.shape
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    nr = row_sums.sum()
    nc = col_sums.sum()

    out = np.zeros((r, c))
    x = np.where(
        np.isin(col_sums, row_sums),
        np.argmax(np.isin(row_sums, col_sums), axis=0) + 1,
        np.nan,
    )
    y = np.where(
        np.isin(row_sums, col_sums),
        np.argmax(np.isin(col_sums, row_sums), axis=0) + 1,
        np.nan,
    )
    x = x[~np.isnan(x)].astype(int) - 1
    y = y[~np.isnan(y)].astype(int) - 1

    out[x, y] = row_sums[x]
    row_sums = row_sums.copy()
    col_sums = col_sums.copy()
    row_sums[x] = 0
    col_sums[y] = 0

    while row_sums.sum() > 0 and col_sums.sum() > 0:
        xi = np.argmax(row_sums)
        yi = np.argmax(col_sums)
        z = min(row_sums[xi], col_sums[yi])
        out[xi, yi] = z
        row_sums[xi] -= z
        col_sums[yi] -= z

    return out


def _goodman_kruskal_lambda(matrix, confidence_level):
    """Compute Goodman-Kruskal Lambda association measure.

    Parameters
    ----------
    matrix : numpy.ndarray
        Contingency table.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    csum = np.sum(matrix, axis=0)
    rsum = np.sum(matrix, axis=1)
    n = np.sum(matrix)
    nrc = np.sum(np.max(matrix, axis=1))
    nkc = np.sum(np.max(matrix, axis=0))
    nrm = np.max(rsum)
    nkm = np.max(csum)

    lambda_row = (nrc - nkm) / (n - nkm)
    lambda_col = (nkc - nrm) / (n - nrm)
    lambda_sym = (nrc + nkc - nrm - nkm) / (2 * n - nrm - nkm)

    # Standard errors (Hartwig, 1976)
    rows_with_max = np.where(rsum == nrm)[0]
    largest_rows = np.array([np.max(matrix[ri, :]) for ri in rows_with_max])
    cols_with_max = np.where(csum == nkm)[0]
    largest_cols = np.array([np.max(matrix[:, ci]) for ci in cols_with_max])

    combos = list(product(largest_rows, largest_cols))
    nrs_tag_vec = np.array([c[0] for c in combos])
    nks_tag_vec = np.array([c[1] for c in combos])

    term1_row = np.sum(np.max(matrix, axis=1) * (np.max(matrix, axis=1) - 1))
    term2_row = np.sum(np.max(matrix, axis=0) * (np.max(matrix, axis=0) - 1))

    ase_row_num = np.sqrt(
        np.sum(
            nrs_tag_vec * (n - nrc) ** 2
            + (nrc - nks_tag_vec) ** 2 * (n - nkm)
            - 2 * nrs_tag_vec * (nrc - nks_tag_vec) * (n - nrc - nkm + nkc)
        )
        / (n * (n - nkm) ** 3)
    )
    ase_col_num = np.sqrt(
        np.sum(
            nks_tag_vec * (n - nkc) ** 2
            + (nkc - nrs_tag_vec) ** 2 * (n - nrm)
            - 2 * nks_tag_vec * (nkc - nrs_tag_vec) * (n - nkc - nrm + nrc)
        )
        / (n * (n - nrm) ** 3)
    )

    ase_row = ase_row_num if ase_row_num > 0 else 1e-10
    ase_col = ase_col_num if ase_col_num > 0 else 1e-10
    ase_sym = np.sqrt(ase_row ** 2 + ase_col ** 2) / 2

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_row = max(lambda_row - zcrit * ase_row, 0)
    upper_row = min(lambda_row + zcrit * ase_row, 1)
    lower_col = max(lambda_col - zcrit * ase_col, 0)
    upper_col = min(lambda_col + zcrit * ase_col, 1)
    lower_sym = max(lambda_sym - zcrit * ase_sym, 0)
    upper_sym = min(lambda_sym + zcrit * ase_sym, 1)

    z_row = lambda_row / ase_row
    z_col = lambda_col / ase_col
    z_sym = lambda_sym / ase_sym
    p_row = norm.sf(z_row)
    p_col = norm.sf(z_col)
    p_sym = norm.sf(z_sym)

    fmt_p_row = "{:.3f}".format(p_row).lstrip("0") if p_row >= 0.001 else "\033[3mp\033[0m < .001"
    fmt_p_col = "{:.3f}".format(p_col).lstrip("0") if p_col >= 0.001 else "\033[3mp\033[0m < .001"
    fmt_p_sym = "{:.3f}".format(p_sym).lstrip("0") if p_sym >= 0.001 else "\033[3mp\033[0m < .001"

    results = {
        "Lambda (Row)": lambda_row,
        "Lambda (Column)": lambda_col,
        "Lambda (Symmetric)": lambda_sym,
        "ASE Lambda (Row)": ase_row,
        "ASE Lambda (Column)": ase_col,
        "ASE Lambda (Symmetric)": ase_sym,
        "Z Lambda (Row)": z_row,
        "Z Lambda (Column)": z_col,
        "Z Lambda (Symmetric)": z_sym,
        "p-value Lambda (Row)": p_row,
        "p-value Lambda (Column)": p_col,
        "p-value Lambda (Symmetric)": p_sym,
        "CI Lambda (Row)": [lower_row, upper_row],
        "CI Lambda (Column)": [lower_col, upper_col],
        "CI Lambda (Symmetric)": [lower_sym, upper_sym],
        "Statistical Line Lambda (Row)": (
            "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(
                lambda_row,
                "\033[3mp = \033[0m" if p_row >= 0.001 else "",
                fmt_p_row,
                confidence_level * 100,
                round(lower_row, 3),
                round(upper_row, 3),
            )
        ),
        "Statistical Line Lambda (Column)": (
            "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(
                lambda_col,
                "\033[3mp = \033[0m" if p_col >= 0.001 else "",
                fmt_p_col,
                confidence_level * 100,
                round(lower_col, 3),
                round(upper_col, 3),
            )
        ),
        "Statistical Line Lambda (Symmetric)": (
            "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(
                lambda_sym,
                "\033[3mp = \033[0m" if p_sym >= 0.001 else "",
                fmt_p_sym,
                confidence_level * 100,
                round(lower_sym, 3),
                round(upper_sym, 3),
            )
        ),
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _goodman_kruskal_tau(matrix, confidence_level):
    """Compute Goodman-Kruskal Tau association measure (Liebtrau, 1983).

    Parameters
    ----------
    matrix : numpy.ndarray
        Contingency table.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    n = np.sum(matrix)
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    cond_err_cols = n ** 2 - np.sum(col_sums ** 2)
    cond_err_rows = n ** 2 - np.sum(row_sums ** 2)
    mean_rows = cond_err_rows / n ** 2
    mean_cols = cond_err_cols / n ** 2

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

    # Tau for rows
    uncond_err_rows = n ** 2 - n * np.sum((matrix[:, np.newaxis] ** 2) / col_sums[np.newaxis])
    tau_rows = 1 - uncond_err_rows / cond_err_rows
    v = uncond_err_rows / n ** 2
    ase_rows = np.sqrt(
        np.sum(
            (
                matrix
                * (
                    -2 * v * (row_sums[:, np.newaxis] / n)
                    + mean_rows * ((2 * matrix / col_sums) - np.sum((matrix / col_sums) ** 2, axis=0))
                    - (mean_rows * (v + 1) - 2 * v)
                )
                ** 2
            )
            / (n ** 2 * mean_rows ** 4)
        )
    )
    ci_rows_lo = max(tau_rows - zcrit * ase_rows, 0)
    ci_rows_hi = min(tau_rows + zcrit * ase_rows, 1)
    z_rows = tau_rows / ase_rows
    p_rows = norm.sf(z_rows)

    # Tau for columns
    uncond_err_cols = n ** 2 - n * np.sum((matrix ** 2) / row_sums[:, np.newaxis])
    tau_cols = 1 - uncond_err_cols / cond_err_cols
    v2 = uncond_err_cols / n ** 2
    ase_cols = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            term = (
                matrix[i, j]
                * (
                    -2 * v2 * (col_sums[j] / n)
                    + mean_cols * ((2 * matrix[i, j] / row_sums[i]) - np.sum((matrix[i, :] / row_sums[i]) ** 2))
                    - (mean_cols * (v2 + 1) - 2 * v2)
                )
                ** 2
                / (n ** 2 * mean_cols ** 4)
            )
            ase_cols += term
    ase_cols = np.sqrt(ase_cols)
    ci_cols_lo = max(tau_cols - zcrit * ase_cols, 0)
    ci_cols_hi = min(tau_cols + zcrit * ase_cols, 1)
    z_cols = tau_cols / ase_cols
    p_cols = norm.sf(z_cols)

    # Symmetric Tau
    alpha = (n ** 2 - np.sum(row_sums ** 2)) / (
        2 * n ** 2 - np.sum(row_sums ** 2) - np.sum(col_sums ** 2)
    )
    tau_sym = tau_rows * alpha + (1 - alpha) * tau_cols
    ase_sym = ase_rows * alpha + (1 - alpha) * ase_cols
    ci_sym_lo = max(tau_sym - zcrit * ase_sym, 0)
    ci_sym_hi = min(tau_sym + zcrit * ase_sym, 1)
    z_sym = tau_sym / ase_sym
    p_sym = norm.sf(z_sym)

    fmt_pr = "{:.3f}".format(p_rows).lstrip("0") if p_rows >= 0.001 else "\033[3mp\033[0m < .001"
    fmt_pc = "{:.3f}".format(p_cols).lstrip("0") if p_cols >= 0.001 else "\033[3mp\033[0m < .001"
    fmt_ps = "{:.3f}".format(p_sym).lstrip("0") if p_sym >= 0.001 else "\033[3mp\033[0m < .001"

    results = {
        "Goodman Kruskal Tau (Rows)": tau_rows,
        "Goodman Kruskal Tau (Columns)": tau_cols,
        "Goodman Kruskal Tau (Symmetric)": tau_sym,
        "Standard Error Rows": ase_rows,
        "Standard Error Columns": ase_cols,
        "Standard Error Symmetric": ase_sym,
        "Statistic Goodman Kruskal Tau (Rows)": z_rows,
        "Statistic Goodman Kruskal Tau (Columns)": z_cols,
        "Statistic Goodman Kruskal Tau (Symmetric)": z_sym,
        "p-value Goodman Kruskal Tau (Rows)": p_rows,
        "p-value Goodman Kruskal Tau (Columns)": p_cols,
        "p-value Goodman Kruskal Tau (Symmetric)": p_sym,
        "Goodman-Kruskal CI's (Rows)": [ci_rows_lo, ci_rows_hi],
        "Goodman-Kruskal CI's (Columns)": [ci_cols_lo, ci_cols_hi],
        "Goodman-Kruskal CI's (Symmetric)": np.around([ci_sym_lo, ci_sym_hi], 4),
        "Statistical Line Goodman Kruskal Tau Rows": (
            "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(
                tau_rows,
                "\033[3mp = \033[0m" if p_rows >= 0.001 else "",
                fmt_pr,
                confidence_level * 100,
                round(ci_rows_lo, 3),
                round(ci_rows_hi, 3),
            )
        ),
        "Statistical Line Goodman Kruskal Tau Columns": (
            "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(
                tau_cols,
                "\033[3mp = \033[0m" if p_cols >= 0.001 else "",
                fmt_pc,
                confidence_level * 100,
                round(ci_cols_lo, 3),
                round(ci_cols_hi, 3),
            )
        ),
        "Statistical Line Goodman Kruskal Tau Symmetric": (
            "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(
                tau_sym,
                "\033[3mp = \033[0m" if p_sym >= 0.001 else "",
                fmt_ps,
                confidence_level * 100,
                round(ci_sym_lo, 3),
                round(ci_sym_hi, 3),
            )
        ),
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _columns_to_contingency(x, y):
    """Convert two categorical vectors to a contingency table.

    Parameters
    ----------
    x, y : array-like
        Categorical variables of equal length.

    Returns
    -------
    numpy.ndarray or None
        Integer contingency table, or ``None`` on error.
    """
    try:
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        mask = (x != "") & (y != "")
        x = x[mask]
        y = y[mask]
        x_cats, x_num = np.unique(x, return_inverse=True)
        y_cats, y_num = np.unique(y, return_inverse=True)
        ct = np.zeros((len(x_cats), len(y_cats)), dtype=int)
        for i in range(len(x)):
            ct[x_num[i], y_num[i]] += 1
        return ct
    except ValueError as exc:
        print(exc)
        return None


def _multilevel_contingency_tables(ct, confidence_level):
    """Compute a comprehensive set of nominal-by-nominal measures.

    Includes chi-square, likelihood ratio, Cramér's V, Tschuprow's T,
    Pearson's contingency coefficient, phi/Cohen's w, uncertainty coefficient,
    and their bias-corrected and maximum-corrected variants, all with
    asymptotic and non-central CIs.

    Parameters
    ----------
    ct : numpy.ndarray
        Contingency table.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    ct = np.asarray(ct)
    n = np.sum(ct)
    n_rows = ct.shape[0]
    n_cols = ct.shape[1]
    q = min(n_cols, n_rows)

    row_totals = np.sum(ct, axis=1)
    col_totals = np.sum(ct, axis=0)
    expected = np.multiply.outer(row_totals, col_totals) / n

    chi_sq = np.sum((ct - expected) ** 2 / expected)
    df_chi = (n_cols - 1) * (n_rows - 1)
    p_val = chi2.sf(chi_sq, df_chi)

    # Likelihood ratio
    exp_lr = np.outer(row_totals, col_totals) / n
    lr = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            if ct[i, j] != 0 and exp_lr[i, j] != 0:
                lr += ct[i, j] * np.log(ct[i, j] / exp_lr[i, j])
    lr *= 2
    p_lr = 1 - chi2.cdf(lr, df_chi)

    phi_sq = chi_sq / n
    phi = np.sqrt(phi_sq)
    cramer_v = np.sqrt(phi_sq / (q - 1))
    tschuprow = np.sqrt(phi_sq / np.sqrt((n_cols - 1) * (n_rows - 1)))
    cc = np.sqrt(chi_sq / (chi_sq + n))

    # Standard deviations
    prob = ct / n
    prob_sq = prob ** 2
    prob_cu = prob ** 3
    col_marg = np.sum(prob, axis=0)
    row_marg = np.sum(prob, axis=1)
    marg_prod = np.outer(row_marg, col_marg)
    corr_prob = prob_sq / marg_prod
    col_marg_sq = col_marg ** 2
    row_marg_sq = row_marg ** 2
    marg_prod_sq = np.outer(row_marg_sq, col_marg_sq)
    corr_prob_sq = prob_cu / marg_prod_sq
    row_marg_c = np.sum(corr_prob, axis=1)
    col_marg_c = np.sum(corr_prob, axis=0)
    marg_prod_c = np.outer(row_marg_c, col_marg_c)
    t1 = np.sum(corr_prob_sq)
    t2 = np.sum(row_marg_c ** 2 / row_marg)
    t3 = np.sum(col_marg_c ** 2 / col_marg)
    t4 = np.sum(marg_prod_c)
    sd_phi_sq = np.sqrt((4 * t1 - 3 * t2 - 3 * t3 + 2 * t4) / n)
    sd_cramer = (1 / (2 * (q - 1) ** 0.5 * cramer_v)) * sd_phi_sq if cramer_v > 0 else 0.0
    sd_tschuprow = (
        sd_phi_sq / np.sqrt(4 * tschuprow ** 2 * np.sqrt((n_cols - 1) * (n_rows - 1)))
    ) if tschuprow > 0 else 0.0
    sd_cc = (1 / (2 * phi * (1 + phi_sq) ** 1.5)) * sd_phi_sq if phi > 0 else 0.0

    # Bias-corrected (Bergsma, 2013)
    phi_sq_bc = max(phi_sq - (1 / (n - 1)) * (n_cols - 1) * (n_rows - 1), 0)
    chi_sq_bc = phi_sq_bc * n
    n_rows_c = max(n_rows - (1 / (n - 1)) * (n_rows - 1) ** 2, 0)
    n_cols_c = max(n_cols - (1 / (n - 1)) * (n_cols - 1) ** 2, 0)
    cramer_bc = np.sqrt(phi_sq_bc / (min(n_cols_c - 1, n_rows_c - 1))) if min(n_cols_c, n_rows_c) > 1 else 0.0
    tschuprow_bc = (
        np.sqrt(phi_sq_bc / np.sqrt((n_cols_c - 1) * (n_rows_c - 1)))
        if n_cols_c > 1 and n_rows_c > 1
        else 0.0
    )
    cc_bc = np.sqrt(phi_sq_bc / (phi_sq_bc + 1)) if phi_sq_bc > 0 else 0.0
    sd_cramer_bc = (1 / (2 * (min(n_cols_c - 1, n_rows_c - 1)) ** 0.5 * cramer_bc)) * sd_phi_sq if cramer_bc > 0 else 0.0
    sd_tschuprow_bc = (
        (1 / (2 * (n_cols_c - 1) * (n_rows_c - 1) * tschuprow_bc)) * sd_phi_sq
        if tschuprow_bc > 0
        else 0.0
    )
    sd_cc_bc = (
        (1 / (2 * np.sqrt(phi_sq_bc) * (1 + phi_sq_bc) ** 1.5)) * sd_phi_sq
        if phi_sq_bc > 0
        else 0.0
    )

    # Maximum-corrected (Berry-Mielke-Johnston)
    max_matrix = _berry_mielke_max_corrected_matrix(np.array(ct))
    obs_flat = max_matrix.flatten()
    exp_max = (np.outer(max_matrix.sum(axis=1), max_matrix.sum(axis=0)) / max_matrix.sum()).flatten()
    zero_pos = np.where((obs_flat == 0) & (exp_max == 0))[0]
    obs_flat = np.delete(obs_flat, zero_pos)
    exp_max = np.delete(exp_max, zero_pos)
    chi_sq_max_bmj = np.sum((obs_flat - exp_max) ** 2 / exp_max)
    cramer_max_bmj = (
        "Maximum Corrected Cramer V is not valid for this sample"
        if chi_sq_max_bmj == 0
        else chi_sq / chi_sq_max_bmj
    )
    max_q = min(n_cols, n_rows)
    chi_sq_max = n * (max_q - 1)
    max_cc = np.sqrt((max_q - 1) / max_q)
    max_t = ((max_q - 1) / max(n_cols - 1, n_rows - 1)) ** 0.25
    tschuprow_max = tschuprow / max_t
    cc_max = cc / max_cc
    se_cc_max = np.sqrt(max_q / (max_q - 1)) * sd_cc
    se_t_max = (max(n_cols - 1, n_rows - 1) / (max_q - 1)) ** 0.25 * sd_tschuprow

    # Asymptotic CIs
    z_crit = norm.ppf(confidence_level + (1 - confidence_level) / 2)
    ci_phi_lo = phi - sd_phi_sq * z_crit
    ci_phi_hi = phi + sd_phi_sq * z_crit
    ci_cramer_lo = max(cramer_v - sd_cramer * z_crit, 0)
    ci_cramer_hi = min(cramer_v + sd_cramer * z_crit, 1)
    ci_t_lo = max(tschuprow - sd_tschuprow * z_crit, 0)
    ci_t_hi = min(tschuprow + sd_tschuprow * z_crit, 1)
    ci_cc_lo = max(cc - sd_cc * z_crit, 0)
    ci_cc_hi = min(cc + sd_cc * z_crit, 1)
    ci_cramer_bc_lo = max(cramer_bc - sd_cramer_bc * z_crit, 0)
    ci_cramer_bc_hi = min(cramer_bc + sd_cramer_bc * z_crit, 1)
    ci_t_bc_lo = max(tschuprow_bc - sd_tschuprow_bc * z_crit, 0)
    ci_t_bc_hi = min(tschuprow_bc + sd_tschuprow_bc * z_crit, 1)
    ci_cc_bc_lo = max(cc_bc - sd_cc_bc * z_crit, 0)
    ci_cc_bc_hi = min(cc_bc + sd_cc_bc * z_crit, 1)
    ci_t_max_lo = max(tschuprow_max - se_t_max * z_crit, 0)
    ci_t_max_hi = min(tschuprow_max + se_t_max * z_crit, 1)
    ci_cc_max_lo = max(cc_max - se_cc_max * z_crit, 0)
    ci_cc_max_hi = min(cc_max + se_cc_max * z_crit, 1)

    # Non-central CIs
    lower_ncp, upper_ncp = _ncp_ci(chi_sq, df_chi, confidence_level)
    lower_ncp = max(lower_ncp, 0)
    lower_ncp_bc, upper_ncp_bc = _ncp_ci(chi_sq_bc, df_chi, confidence_level)
    lower_ncp_max, upper_ncp_max = _ncp_ci(chi_sq_max, df_chi, confidence_level)
    lower_ncp_max = max(lower_ncp_max, 0)
    lower_ncp_bmj, upper_ncp_bmj = (
        _ncp_ci(chi_sq_max_bmj, df_chi, confidence_level)
        if chi_sq_max_bmj > 0
        else (0, 0)
    )
    lower_ncp_bmj = max(lower_ncp_bmj, 0)

    lower_phi_sq = max(lower_ncp / n, 0)
    upper_phi_sq = min(upper_ncp / n, 1)
    lower_phi_sq_bc = 0 if lower_ncp_bc == 0 else lower_ncp_bc / n
    upper_phi_sq_bc = 0 if upper_ncp_bc == 0 else upper_ncp_bc / n
    lower_phi_sq_max = max(lower_ncp_max / n, 0)
    upper_phi_sq_max = min(upper_ncp_max / n, 1)

    ci_ncp_cramer_lo = max(np.sqrt(lower_phi_sq / (q - 1)), 0)
    ci_ncp_cramer_hi = min(np.sqrt(upper_phi_sq / (q - 1)), 1)
    ci_ncp_t_lo = max(np.sqrt(lower_phi_sq / np.sqrt(df_chi)), 0)
    ci_ncp_t_hi = min(np.sqrt(upper_phi_sq / np.sqrt(df_chi)), 1)
    ci_ncp_cc_lo = max(np.sqrt(lower_phi_sq / (lower_phi_sq + 1)), 0)
    ci_ncp_cc_hi = min(np.sqrt(upper_phi_sq / (upper_phi_sq + 1)), 1)
    ci_ncp_cramer_bc_lo = max(np.sqrt(lower_phi_sq_bc / (q - 1)), 0)
    ci_ncp_cramer_bc_hi = np.sqrt(upper_phi_sq_bc / (q - 1))
    ci_ncp_t_bc_lo = max(np.sqrt(lower_phi_sq_bc / np.sqrt(df_chi)), 0)
    ci_ncp_t_bc_hi = np.sqrt(upper_phi_sq_bc / np.sqrt(df_chi))
    ci_ncp_cc_bc_lo = max(np.sqrt(lower_phi_sq_bc / (lower_phi_sq_bc + 1)), 0)
    ci_ncp_cc_bc_hi = np.sqrt(upper_phi_sq_bc / (upper_phi_sq_bc + 1))
    ci_ncp_cramer_max_lo = (
        0
        if chi_sq_max_bmj == 0
        else max(lower_ncp / lower_ncp_bmj, 0)
        if lower_ncp_bmj > 0
        else 0
    )
    ci_ncp_cramer_max_hi = (
        0 if chi_sq_max_bmj == 0 else upper_ncp / upper_ncp_bmj if upper_ncp_bmj > 0 else 0
    )

    # Uncertainty coefficient (Theil's U)
    row_s = np.sum(ct, axis=1)
    col_s = np.sum(ct, axis=0)
    hx = np.sum((row_s * np.log(row_s / n)) / n)
    hy = np.sum((col_s * np.log(col_s / n)) / n)
    pij = ct / n
    term1 = np.where(pij == 0, 0, np.log(pij + (pij == 0)))
    term2 = np.outer(row_s, col_s) / n ** 2
    term3 = np.where(term2 == 0, 0, np.log(term2))
    term4 = row_s / n
    term5 = np.where(term4 == 0, 0, np.log(term4))
    term6 = ct / col_s
    term7 = np.where(term6 == 0, 0, np.where(term6 == 1, 0, np.log(term6 + (term6 == 0))))
    term8 = col_s / n
    term9 = np.where(term8 == 0, 0, np.log(term8))
    term10 = ct / row_s[:, np.newaxis]
    term11 = np.where(term10 == 0, 0, np.where(term10 == 1, 0, np.log(term10 + (term10 == 0))))
    hxy = np.sum(ct * term1) / n

    uc_sym = 2 * (hx + hy - hxy) / (hx + hy)
    uc_rows = (hx + hy - hxy) / hx
    uc_cols = (hx + hy - hxy) / hy
    se_uc_sym = np.sqrt(
        4 * np.sum(ct * (hxy * term3 - (hx + hy) * term1) ** 2) / (n ** 2 * (hx + hy) ** 4)
    )
    se_uc_rows = np.sqrt(
        np.sum(ct * (hx * term7 + (hy - hxy) * term5[:, np.newaxis]) ** 2) / (n ** 2 * hx ** 4)
    )
    se_uc_cols = np.sqrt(
        np.sum(ct * (hy * term11 + (hx - hxy) * term9) ** 2) / (n ** 2 * hy ** 4)
    )
    z_uc_sym = "inf" if uc_sym == 1 else uc_sym / se_uc_sym
    z_uc_rows = "inf" if uc_rows == 1 else uc_rows / se_uc_rows
    z_uc_cols = "inf" if uc_cols == 1 else uc_cols / se_uc_cols
    uc_zcrit = 1 - (1 - confidence_level) / 2
    ci_uc_sym_lo = max(uc_sym - uc_zcrit * np.sqrt(se_uc_sym), 0)
    ci_uc_sym_hi = min(uc_sym + uc_zcrit * np.sqrt(se_uc_sym), 1)
    ci_uc_rows_lo = max(uc_rows - uc_zcrit * np.sqrt(se_uc_rows), 0)
    ci_uc_rows_hi = min(uc_rows + uc_zcrit * np.sqrt(se_uc_rows), 1)
    ci_uc_cols_lo = max(uc_cols - uc_zcrit * np.sqrt(se_uc_cols), 0)
    ci_uc_cols_hi = min(uc_cols + uc_zcrit * np.sqrt(se_uc_cols), 1)

    fmt_p = "{:.3f}".format(p_val).lstrip("0") if p_val >= 0.001 else "\033[3mp\033[0m < .001"

    results = {
        "Confidence_Level": round(confidence_level, 4),
        "chi_square_Bias_corrected": round(chi_sq_bc, 4),
        "Chi Square": round(chi_sq, 4),
        "Degrees of Freedom Chi Square": round(df_chi, 4),
        "p_value Chi Square": p_val,
        "Likelihood Ratio": np.around(lr, 4),
        "Likelihood Ratio p_value": p_lr,
        "___________________________________________": "",
        "Cramer V": round(cramer_v, 4),
        "Pearson's Contingency Coefficient": round(cc, 4),
        "Tschuprow's T": round(tschuprow, 7),
        "Standard Error of Cramer V": round(sd_cramer, 4),
        "Standard Error of Contingency Coefficient": round(sd_cc, 4),
        "Standard Error of Tschuprow's T": round(sd_tschuprow, 4),
        "Asymptotic CI Cramer V": f"({round(ci_cramer_lo, 4)}, {round(ci_cramer_hi, 4)})",
        "Asymptotic CI Contingency Coefficient": f"({round(ci_cc_lo, 4)}, {round(ci_cc_hi, 4)})",
        "Asymptotic Tschuprow's T": f"({round(ci_t_lo, 4)}, {round(ci_t_hi, 4)})",
        "NCP CI Cramer V": f"({round(ci_ncp_cramer_lo, 4)}, {round(ci_ncp_cramer_hi, 4)})",
        "NCP CI Contingency Coefficient": f"({round(ci_ncp_cc_lo, 4)}, {round(ci_ncp_cc_hi, 4)})",
        "NCP CI Tschuprow's T": f"({round(ci_ncp_t_lo, 4)}, {round(ci_ncp_t_hi, 4)})",
        "Statistical Line Cramer's V": (
            "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cramer's \033[3mV\033[0m = {:.3f}, "
            "{}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                int(df_chi), n, chi_sq,
                "\033[3mp = \033[0m" if p_val >= 0.001 else "", fmt_p,
                round(cramer_v, 3), confidence_level * 100,
                round(ci_ncp_cramer_lo, 3), round(ci_ncp_cramer_hi, 3),
            )
        ),
        "____________________________________________": "",
        "Adjusted Cramer's V": round(cramer_bc, 7),
        "Adjusted Tschuprow's T": round(tschuprow_bc, 7),
        "Adjusted Contingency Coefficient": round(cc_bc, 4),
        "Standard Error of Bias Corrected Cramer's V": round(sd_cramer_bc, 4),
        "Standard Error of Bias Corrected Tschuprows T": round(sd_tschuprow_bc, 4),
        "Standard Error of Bias Corrected Contingency Coefficient": round(sd_cc_bc, 4),
        "CI bias corrected Cramer V": f"({np.around(ci_cramer_bc_lo, 4)}, {np.around(ci_cramer_bc_hi, 4)})",
        "CI bias corrected Tschuprows T": f"({np.around(ci_t_bc_lo, 4)}, {np.around(ci_t_bc_hi, 4)})",
        "CI bias corrected Contingency Coefficient": f"({np.around(ci_cc_bc_lo, 4)}, {np.around(ci_cc_bc_hi, 4)})",
        "NCP CI Cramer's V Bias Corrected": f"({np.around(ci_ncp_cramer_bc_lo, 4)}, {np.around(ci_ncp_cramer_bc_hi, 4)})",
        "NCP CI Tschuprow's T Bias Corrected": f"({np.around(ci_ncp_t_bc_lo, 4)}, {np.around(ci_ncp_t_bc_hi, 4)})",
        "NCP CI Contingency Coefficient Bias Corrected": f"({np.around(ci_ncp_cc_bc_lo, 4)}, {np.around(ci_ncp_cc_bc_hi, 4)})",
        "______________________________________________": "",
        "Maximum Corrected Cramers V (Berry, Mielke, Johnston)": cramer_max_bmj,
        "Maximum Corrected Tschuprow's T": round(tschuprow_max, 7),
        "Maximum Corrected Contingency Coefficient (Sakoda, 1977)": round(cc_max, 4),
        "Standard Error of Maximum Corrected Tschuprows T": round(se_t_max, 4),
        "Standard Error of Maximum Corrected Contingency Coefficient": round(se_cc_max, 4),
        "CI Max corrected Tschuprows T": f"({np.around(ci_t_max_lo, 4)}, {np.around(ci_t_max_hi, 4)})",
        "CI Max corrected Contingency Coefficient": f"({np.around(ci_cc_max_lo, 4)}, {np.around(ci_cc_max_hi, 4)})",
        "NCP CI Cramer's V Max corrected": f"({round(ci_ncp_cramer_max_lo, 4)}, {np.around(ci_ncp_cramer_max_hi, 4)})",
        "__________________________________________________": "",
        "Phi / Cohen's w": round(phi, 4),
        "Standard Deviation of Phi Square": round(sd_phi_sq, 4),
        "Adjusted Phi": round(max(phi_sq_bc, 0), 7),
        "Asymptotic CI Phi / Cohens w": f"({round(ci_phi_lo, 4)}, {round(ci_phi_hi, 4)})",
        "NCP CI's Phi / Cohens w": f"({round(np.sqrt(lower_phi_sq), 4)}, {round(np.sqrt(upper_phi_sq), 4)})",
        "Ncp CI's Adjusted Phi / Cohens w": (
            f"({np.around(np.sqrt(lower_phi_sq_bc), 4)}, {np.around(np.sqrt(upper_phi_sq_bc), 4)})"
        ),
        "Statistical Line Cohen's w": (
            "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cohen's \033[3mw\033[0m = {:.3f}, "
            "{}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                int(df_chi), n, chi_sq,
                "\033[3mp = \033[0m" if p_val >= 0.001 else "", fmt_p,
                round(phi, 3), confidence_level * 100,
                round(np.sqrt(lower_phi_sq), 3), round(np.sqrt(upper_phi_sq), 3),
            )
        ),
        "Theil's Uncertainty Coefficient (Symmetric)": uc_sym,
        "Theil's Uncertainty Coefficient (Rows)": uc_rows,
        "Theil's Uncertainty Coefficient (Columns)": uc_cols,
        "Theil's Uncertainty Coefficient Standard Error (Symmetric)": se_uc_sym,
        "Theil's Uncertainty Coefficient Standard Error (Rows)": se_uc_rows,
        "Theil's Uncertainty Coefficient Standard Error (Columns)": se_uc_cols,
        "Theil's Uncertainty Coefficient Z-value (Symmetric)": z_uc_sym,
        "Theil's Uncertainty Coefficient Z-value (Rows)": z_uc_rows,
        "Theil's Uncertainty Coefficient Z-value (Columns)": z_uc_cols,
        "Theil's Uncertainty Coefficient Confidence Intervals (Symmetric)": (
            f"({round(ci_uc_sym_lo, 4)}, {round(ci_uc_sym_hi, 4)})"
        ),
        "Theil's Uncertainty Coefficient Confidence Intervals (Rows)": (
            f"({round(ci_uc_rows_lo, 4)}, {round(ci_uc_rows_hi, 4)})"
        ),
        "Theil's Uncertainty Coefficient Confidence Intervals (Columns)": (
            f"({round(ci_uc_cols_lo, 4)}, {round(ci_uc_cols_hi, 4)})"
        ),
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class NominalByNominal:
    """Nominal by Nominal association measures.

    Computes Cohen's w, Cramér's V, contingency coefficient, Goodman-Kruskal
    Lambda, Goodman-Kruskal Tau, and many related measures from a chi-square
    score, a contingency table, or raw data vectors.
    """

    @staticmethod
    def from_chi_score(params: dict) -> dict:
        """Calculate measures from an observed chi-square score.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Chi Square Score"`` : float, observed chi-square statistic.
            - ``"Sample Size"`` : int, total sample size.
            - ``"Degrees of Freedom"`` : int.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).

        Returns
        -------
        dict
            Dictionary of effect sizes and confidence intervals.
        """
        chi_sq = params["Chi Square Score"]
        n = params["Sample Size"]
        df = params["Degrees of Freedom"]
        cl_pct = params["Confidence Level"]
        cl = cl_pct / 100

        cohens_w = np.sqrt(chi_sq / n)
        cramer_v = np.sqrt(chi_sq / (df * n))
        cont_coef = np.sqrt(chi_sq / (chi_sq + n))

        lower_chi, upper_chi = _ncp_ci(chi_sq, df, cl)
        lower_w = np.sqrt(lower_chi / n)
        upper_w = np.sqrt(upper_chi / n)
        lower_cramer = np.sqrt(lower_chi / (df * n))
        upper_cramer = np.sqrt(upper_chi / (df * n))
        lower_cc = np.sqrt(lower_chi / (lower_chi + n))
        upper_cc = np.sqrt(upper_chi / (upper_chi + n))
        p_val = chi2.sf(abs(chi_sq), df)

        fmt_p = "{:.3f}".format(p_val).lstrip("0") if p_val >= 0.001 else "\033[3mp\033[0m < .001"
        results = {
            "Cohen's w / Phi": round(cohens_w, 4),
            "Cramer's V": round(cramer_v, 4),
            "Contingency Coefficient": round(cont_coef, 4),
            "Chi Square Score": round(chi_sq, 4),
            "Degrees of Freedom": round(df, 4),
            "p-value": np.around(p_val, 4),
            "Cohen's w CI Lower": round(lower_w, 4),
            "Cohen's w CI Upper": round(upper_w, 4),
            "Cramer's V CI Lower": round(lower_cramer, 4),
            "Cramer's V CI Upper": round(upper_cramer, 4),
            "Contingency Coefficient CI Lower": round(lower_cc, 4),
            "Contingency Coefficient CI Upper": round(upper_cc, 4),
            "Statistical Line Cohen's w": (
                "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cohen's \033[3mw\033[0m = {:.3f}, "
                "{}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                    int(df), n, chi_sq,
                    "\033[3mp = \033[0m" if p_val >= 0.001 else "", fmt_p,
                    round(cohens_w, 3), cl_pct,
                    round(lower_w, 3), round(upper_w, 3),
                )
            ),
            "Statistical Line Cramer's V": (
                "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cramer's \033[3mV\033[0m = {:.3f}, "
                "{}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                    int(df), n, chi_sq,
                    "\033[3mp = \033[0m" if p_val >= 0.001 else "", fmt_p,
                    round(cramer_v, 3), cl_pct,
                    round(lower_cramer, 3), round(upper_cramer, 3),
                )
            ),
            "Statistical Line Contingency Coefficient": (
                "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, \033[3mC\033[0m = {:.3f}, "
                "{}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                    int(df), n, chi_sq,
                    "\033[3mp = \033[0m" if p_val >= 0.001 else "", fmt_p,
                    round(cont_coef, 3), cl_pct,
                    round(lower_cc, 3), round(upper_cc, 3),
                )
            ),
        }
        return results

    @staticmethod
    def from_contingency_table(params: dict) -> dict:
        """Calculate measures from a contingency table.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Contingency Table"`` : 2-D numpy array.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).

        Returns
        -------
        dict
            Dictionary with multilevel contingency table output, Lambda table,
            and Tau table.
        """
        ct = np.array(params["Contingency Table"])
        cl_pct = params["Confidence Level"]
        cl = cl_pct / 100

        out_multilevel = _multilevel_contingency_tables(ct, cl)
        out_lambda = _goodman_kruskal_lambda(ct, cl)
        out_tau = _goodman_kruskal_tau(ct, cl)

        return {
            "Nominal by Nominal Association": out_multilevel,
            "Lambda Table": out_lambda,
            "Tau Table": out_tau,
        }

    @staticmethod
    def from_data(params: dict) -> dict:
        """Calculate measures from raw data vectors.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Column 1"`` : array-like, first categorical variable.
            - ``"Column 2"`` : array-like, second categorical variable.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).

        Returns
        -------
        dict
            Dictionary with the contingency table, Lambda table, and Tau table.
        """
        col1 = params["Column 1"]
        col2 = params["Column 2"]
        cl_pct = params["Confidence Level"]
        cl = cl_pct / 100

        ct = _columns_to_contingency(col1, col2)
        out_lambda = _goodman_kruskal_lambda(ct, cl)
        out_tau = _goodman_kruskal_tau(ct, cl)

        return {
            "Contingency Table": ct,
            "Lambda Table": out_lambda,
            "Tau Table": out_tau,
        }
