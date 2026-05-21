"""Module for Nominal by Ordinal correlation measures."""

import math

import numpy as np
import pandas as pd
from scipy.stats import kruskal as scipy_kruskal
from scipy.stats import norm


def _freemans_theta(contingency_table):
    """Compute Freeman's Theta from a contingency table.

    Parameters
    ----------
    contingency_table : array-like or DataFrame
        Contingency table with rows as nominal categories and columns as
        ordered ordinal categories.

    Returns
    -------
    float
        Freeman's Theta statistic.
    """
    ct = pd.DataFrame(contingency_table)
    row_names = ct.index
    vectors = {}
    contrasts = {}

    for i in range(len(row_names)):
        for j in range(i + 1, len(row_names)):
            row1 = ct.loc[row_names[i]].values
            row2 = ct.loc[row_names[j]].values
            vectors[f"{row_names[i]}_{row_names[j]}"] = np.multiply.outer(row1, row2)

    for i in range(len(row_names)):
        for j in range(i + 1, len(row_names)):
            v = vectors[f"{row_names[i]}_{row_names[j]}"]
            contrasts[f"{row_names[i]}_{row_names[j]}"] = np.sum(np.triu(v)) - np.sum(np.tril(v))

    delta = np.sum(np.abs(list(contrasts.values())))
    row_sums = ct.sum(axis=1).values
    t2 = np.sum(np.triu(np.outer(row_sums, row_sums), k=1))
    return delta / t2


def _rank_biserial_correlation(nominal, ordinal, confidence_level):
    """Compute rank-biserial correlation for a binary nominal variable.

    Parameters
    ----------
    nominal : array-like
        Binary nominal variable.
    ordinal : array-like
        Ordinal variable.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    tuple[float, float, float] or str
        ``(rbc, lower_ci, upper_ci)`` or an error message if the nominal
        variable is not binary.
    """
    unique_vals = set(nominal)
    if len(unique_vals) != 2:
        return "There are more than two values; rank-biserial correlation requires a binary nominal variable."

    val_map = {v: idx for idx, v in enumerate(unique_vals)}
    binary = [val_map[item] for item in nominal]

    count_a_gt_b = sum(
        1 for a, b in zip(binary, ordinal) if a == 0
        for a2, b2 in zip(binary, ordinal) if a2 == 1 and b < b2
    )
    count_b_gt_a = sum(
        1 for a, b in zip(binary, ordinal) if a == 0
        for a2, b2 in zip(binary, ordinal) if a2 == 1 and b > b2
    )
    n1 = binary.count(0)
    n2 = binary.count(1)
    n_comparisons = n1 * n2

    rbc = ((count_a_gt_b / n_comparisons) - (count_b_gt_a / n_comparisons)) / 2
    se = np.sqrt((n1 + n2 + 1) / (3 * n1 + n2))
    z_crit = norm.ppf((1 - confidence_level) + (confidence_level / 2))
    lower = max(math.tanh(math.atanh(rbc) - z_crit * se), -1)
    upper = min(math.tanh(math.atanh(rbc) + z_crit * se), 1)
    return rbc, lower, upper


def _kruskal_wallis(groups):
    """Run Kruskal-Wallis test using scipy.

    Parameters
    ----------
    groups : list of array-like
        Data groups to compare.

    Returns
    -------
    tuple[float, float, float]
        ``(df, H_statistic, p_value)``
    """
    h_stat, p_val = scipy_kruskal(*groups)
    df_kw = len(groups) - 1
    return df_kw, h_stat, p_val


class NominalByOrdinal:
    """Nominal by Ordinal correlation measures.

    Computes epsilon-squared (Kruskal-Wallis effect size), Freeman's Theta,
    and rank-biserial correlation (when nominal has exactly two levels).
    Confidence intervals are obtained via bootstrapping.
    """

    @staticmethod
    def from_contingency_table(params: dict) -> dict:
        """Calculate measures from a contingency table.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Contingency Table"`` : 2-D numpy array.
            - ``"Number of Bootstrapping Samples"`` : int.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).

        Returns
        -------
        dict
            Result dictionary containing H statistic, degrees of freedom,
            p-value, epsilon-squared with bootstrap CIs, Freeman's Theta
            with bootstrap CIs, and (if binary nominal) rank-biserial
            correlation with CIs.
        """
        table = np.array(params["Contingency Table"])
        n_boots = int(params["Number of Bootstrapping Samples"])
        cl_pct = float(params["Confidence Level"])
        cl = cl_pct / 100

        # Build long-format data frame from the contingency table
        df = pd.DataFrame(
            {
                "Group": np.repeat(np.arange(table.shape[0]), table.shape[1]),
                "Response": table.flatten(),
            }
        )
        df["Group"] = np.array([chr(97 + g) for g in df["Group"]])
        df["Cumulative Frequency"] = df.groupby("Group").cumcount() + 1
        df = pd.DataFrame(
            df[["Group", "Cumulative Frequency"]].values.repeat(df["Response"], axis=0),
            columns=["Nominal_Variable", "Ordinal_Variable"],
        )
        df["Ordinal_Variable"] = pd.to_numeric(df["Ordinal_Variable"])

        n = len(df["Ordinal_Variable"])
        n_levels = len(np.unique(df["Nominal_Variable"]))

        # 1. Kruskal-Wallis test and epsilon-squared
        groups = [df.loc[df["Nominal_Variable"] == lv, "Ordinal_Variable"].values
                  for lv in np.unique(df["Nominal_Variable"])]
        df_kw, h_stat, p_kw = _kruskal_wallis(groups)
        epsilon_sq = h_stat / (n - 1)

        # Bootstrap CI for epsilon-squared
        boot_h = []
        for _ in range(n_boots):
            sample = df.sample(frac=1, replace=True)
            boot_groups = [
                sample.loc[sample["Nominal_Variable"] == lv, "Ordinal_Variable"].values
                for lv in np.unique(sample["Nominal_Variable"])
            ]
            try:
                _, bh, _ = _kruskal_wallis(boot_groups)
                boot_h.append(bh)
            except Exception:
                pass

        upper_h = np.percentile(boot_h, 100 - (100 - cl_pct) / 2)
        lower_h = np.percentile(boot_h, (100 - cl_pct) / 2)
        epsilon_upper = upper_h / (n - 1)
        epsilon_lower = lower_h / (n - 1)

        # 2. Freeman's Theta and bootstrap CI
        freemans = _freemans_theta(table)
        theta_boots = []
        for _ in range(n_boots):
            sample = df.sample(frac=1, replace=True).reset_index(drop=True)
            ct = pd.crosstab(sample["Nominal_Variable"], sample["Ordinal_Variable"])
            try:
                theta_boots.append(_freemans_theta(ct))
            except Exception:
                pass
        theta_lower = np.percentile(theta_boots, (100 - cl_pct) / 2)
        theta_upper = np.percentile(theta_boots, 100 - (100 - cl_pct) / 2)

        results = {
            "H Statistic": h_stat,
            "Degrees of Freedom of the Kruskal Wallis Test": df_kw,
            "p-value of the Kruskal Wallis Test": p_kw,
            "Epsilon Square": epsilon_sq,
            "Lower Ci of Epsilon Square": epsilon_lower,
            "Upper Ci of Epsilon Square": epsilon_upper,
            "Freeman's Theta": freemans,
            "Freeman's Theta Lower CI": theta_lower,
            "Freeman's Theta Upper Ci": theta_upper,
        }

        fmt_p = "{:.3f}".format(p_kw).lstrip("0") if p_kw >= 0.001 else "\033[3mp\033[0m < .001"
        h_fmt = int(round(h_stat, 3)) if float(round(h_stat, 3)).is_integer() else round(h_stat, 3)
        results["Statistical Line Epsilon Square"] = (
            " \033[3mH\033[0m({}) = {}, {}{}, \033[3m\u03B5\u00B2\033[0m = {}, {}% CI(bootstrapping) [{}, {}]".format(
                df_kw,
                h_fmt,
                "\033[3mp = \033[0m" if p_kw >= 0.001 else "",
                fmt_p,
                ("-" if str(epsilon_sq).startswith("-") else "") + str(round(epsilon_sq, 3)).lstrip("-").lstrip("0") or "0",
                cl_pct,
                ("-" if str(epsilon_lower).startswith("-") else "") + str(round(epsilon_lower, 3)).lstrip("-").lstrip("0") or "0",
                ("-" if str(epsilon_upper).startswith("-") else "") + str(round(epsilon_upper, 3)).lstrip("-").lstrip("0") or "0",
            )
        )

        # 3. Rank-biserial correlation (binary nominal only)
        if n_levels == 2:
            rbc_result = _rank_biserial_correlation(
                df["Nominal_Variable"].tolist(), df["Ordinal_Variable"].tolist(), cl
            )
            if isinstance(rbc_result, tuple):
                rbc, rbc_lower, rbc_upper = rbc_result
                results["Rank Biserial Correlation"] = rbc
                results["Lower CI Rank Biserial Correlation"] = rbc_lower
                results["Upper CI Rank Biserial Correlation"] = rbc_upper

        return results

    @staticmethod
    def from_data(params: dict) -> dict:
        """Calculate measures from raw data vectors.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Nominal Variable"`` : array-like, nominal grouping variable.
            - ``"Ordinal Variable"`` : array-like, ordinal variable.
            - ``"Number of Bootstrapping Samples"`` : int.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).

        Returns
        -------
        dict
            Result dictionary (same structure as :meth:`from_contingency_table`).
        """
        nominal = params["Nominal Variable"]
        ordinal = params["Ordinal Variable"]
        n_boots = int(params["Number of Bootstrapping Samples"])
        cl_pct = float(params["Confidence Level"])
        cl = cl_pct / 100

        df = pd.DataFrame({"Ordinal_Variable": ordinal, "Nominal_Variable": nominal})
        ct = pd.crosstab(df["Nominal_Variable"], df["Ordinal_Variable"])
        n = len(df["Ordinal_Variable"])
        n_levels = len(np.unique(df["Nominal_Variable"]))

        # 1. Kruskal-Wallis
        groups = [df.loc[df["Nominal_Variable"] == lv, "Ordinal_Variable"].values
                  for lv in np.unique(df["Nominal_Variable"])]
        df_kw, h_stat, p_kw = _kruskal_wallis(groups)
        epsilon_sq = h_stat / (n - 1)

        # Bootstrap CI
        boot_h = []
        for _ in range(n_boots):
            sample = df.sample(frac=1, replace=True)
            boot_groups = [
                sample.loc[sample["Nominal_Variable"] == lv, "Ordinal_Variable"].values
                for lv in np.unique(sample["Nominal_Variable"])
            ]
            try:
                _, bh, _ = _kruskal_wallis(boot_groups)
                boot_h.append(bh)
            except Exception:
                pass

        upper_h = np.percentile(boot_h, 100 - (100 - cl_pct) / 2)
        lower_h = np.percentile(boot_h, (100 - cl_pct) / 2)
        epsilon_upper = upper_h / (n - 1)
        epsilon_lower = lower_h / (n - 1)

        # 2. Freeman's Theta
        freemans = _freemans_theta(ct)
        theta_boots = []
        for _ in range(n_boots):
            sample = df.sample(frac=1, replace=True).reset_index(drop=True)
            ct_boot = pd.crosstab(sample["Nominal_Variable"], sample["Ordinal_Variable"])
            try:
                theta_boots.append(_freemans_theta(ct_boot))
            except Exception:
                pass
        theta_lower = np.percentile(theta_boots, (100 - cl_pct) / 2)
        theta_upper = np.percentile(theta_boots, 100 - (100 - cl_pct) / 2)

        results = {
            "H Statistic": h_stat,
            "Degrees of Freedom of the Kruskal Wallis Test": df_kw,
            "p-value of the Kruskal Wallis Test": p_kw,
            "Epsilon Square": epsilon_sq,
            "Lower Ci of Epsilon Square": epsilon_lower,
            "Upper Ci of Epsilon Square": epsilon_upper,
            "Freeman's Theta": freemans,
            "Freeman's Theta Lower CI": theta_lower,
            "Freeman's Theta Upper Ci": theta_upper,
        }

        fmt_p = "{:.3f}".format(p_kw).lstrip("0") if p_kw >= 0.001 else "\033[3mp\033[0m < .001"
        h_fmt = int(round(h_stat, 3)) if float(round(h_stat, 3)).is_integer() else round(h_stat, 3)
        results["Statistical Line Epsilon Square"] = (
            " \033[3mH\033[0m({}) = {}, {}{}, \033[3m\u03B5\u00B2\033[0m = {}, {}% CI(bootstrapping) [{}, {}]".format(
                df_kw,
                h_fmt,
                "\033[3mp = \033[0m" if p_kw >= 0.001 else "",
                fmt_p,
                ("-" if str(epsilon_sq).startswith("-") else "") + str(round(epsilon_sq, 3)).lstrip("-").lstrip("0") or "0",
                cl_pct,
                ("-" if str(epsilon_lower).startswith("-") else "") + str(round(epsilon_lower, 3)).lstrip("-").lstrip("0") or "0",
                ("-" if str(epsilon_upper).startswith("-") else "") + str(round(epsilon_upper, 3)).lstrip("-").lstrip("0") or "0",
            )
        )

        # 3. Rank-biserial correlation (binary nominal only)
        if n_levels == 2:
            rbc_result = _rank_biserial_correlation(
                df["Nominal_Variable"].tolist(), df["Ordinal_Variable"].tolist(), cl
            )
            if isinstance(rbc_result, tuple):
                rbc, rbc_lower, rbc_upper = rbc_result
                results["Rank Biserial Correlation"] = rbc
                results["Lower CI Rank Biserial Correlation"] = rbc_lower
                results["Upper CI Rank Biserial Correlation"] = rbc_upper

        return results
