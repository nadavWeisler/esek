"""Module for Ordinal by Ordinal correlation measures."""

import numpy as np
import pandas as pd
from scipy.stats import norm, weightedtau

from .ordinal_by_interval import (
    _ginis_gamma,
    _gaussian_rank_correlation,
    _shepherd,
    _skipped_correlation,
    _spearman_correlation,
)


def _gamma_family_measures(column_1, column_2, confidence_level):
    """Compute the Gamma family of ordinal association measures.

    Computes Kendall's Tau-a, Tau-b, Stuart's Tau-c, Somers' Delta,
    Goodman-Kruskal Gamma, Wilson's e, and Weighted Tau.

    Parameters
    ----------
    column_1, column_2 : array-like
        Paired ordinal data vectors.
    confidence_level : float
        Confidence level (0–1) for confidence intervals.

    Returns
    -------
    str
        Formatted results string.
    """
    data = pd.DataFrame({"coloumn_1": column_1, "coloumn_2": column_2})
    ct = pd.crosstab(data["coloumn_1"], data["coloumn_2"])
    n = np.sum(np.sum(ct, axis=1))
    n_rows, n_cols = ct.shape
    q = min(n_rows, n_cols)

    final = ct.reset_index().melt(id_vars="coloumn_1", var_name="coloumn_2", value_name="Nij")
    final["Ni"] = final["coloumn_1"].map(data["coloumn_1"].value_counts())
    final["Nj"] = final["coloumn_2"].map(data["coloumn_2"].value_counts())
    final["Concordant Pairs"] = 0
    final["Disconcordant Pairs"] = 0

    for i, row in final.iterrows():
        x_val = row["coloumn_1"]
        y_val = row["coloumn_2"]
        concordant = 0
        discordant = 0
        for _, other in final[final["coloumn_1"] != x_val].iterrows():
            count = other["Nij"]
            if (x_val > other["coloumn_1"] and y_val > other["coloumn_2"]) or (
                x_val < other["coloumn_1"] and y_val < other["coloumn_2"]
            ):
                concordant += count
            if (x_val > other["coloumn_1"] and y_val < other["coloumn_2"]) or (
                x_val < other["coloumn_1"] and y_val > other["coloumn_2"]
            ):
                discordant += count
        final.at[i, "Concordant Pairs"] = concordant
        final.at[i, "Disconcordant Pairs"] = discordant
        final.at[i, "(C-D)^2*Nij"] = (concordant - discordant) ** 2 * row["Nij"]
        final.at[i, "Concordant Pairs * Frequency"] = concordant * row["Nij"]
        final.at[i, "Disconcordant Pairs * Frequency"] = discordant * row["Nij"]
        final.at[i, "P"] = concordant * row["Nij"]
        final.at[i, "Q"] = discordant * row["Nij"]
        final.at[i, "Cij - Dij"] = concordant - discordant
        final.at[i, "vij"] = (
            (n ** 2 - np.sum(np.sum(ct, axis=1) ** 2)) * row["Nj"]
            + (n ** 2 - np.sum(np.sum(ct, axis=0) ** 2)) * row["Ni"]
        )

    P = final["P"].sum()
    Q = final["Q"].sum()
    d_var1 = n ** 2 - np.sum(np.sum(ct, axis=1) ** 2)
    d_var2 = n ** 2 - np.sum(np.sum(ct, axis=0) ** 2)

    tau_a = (P - Q) / (n * (n - 1))
    tau_b = (P - Q) / np.sqrt(d_var2 * d_var1)
    tau_c = q * (P - Q) / (n ** 2 * (q - 1))
    wtau, wtau_p = weightedtau(column_1, column_2)

    ci_cd = final["Cij - Dij"]
    c_bar = np.sum(ci_cd) / n

    ase1_tau_a = np.sqrt(
        2 / (n * (n - 1))
        * (
            (2 * (n - 2)) / (n * (n - 1) ** 2) * sum((ci_cd - c_bar) ** 2)
            + 1
            - tau_a ** 2
        )
    )
    term3 = 1 / (d_var1 * d_var2)
    term4 = 2 * np.sqrt(d_var1 * d_var2)
    term5 = n ** 3 * tau_b ** 2 * (d_var1 + d_var2) ** 2
    ase1_tau_b = term3 * np.sqrt(
        np.sum(
            final["Nij"]
            * (term4 * final["Cij - Dij"] + final["vij"] * tau_b) ** 2
        )
        - term5
    )
    ase1_tau_c = ((2 * q) / ((q - 1) * n ** 2)) * np.sqrt(
        np.sum(final["Nij"] * (final["Disconcordant Pairs"] - final["Concordant Pairs"]) ** 2)
        - (1 / n * (P - Q) ** 2)
    )

    ase0_tau_a = ase1_tau_a
    ase0_tau_b = np.sqrt(
        (
            np.sum(final["Nij"] * (final["Disconcordant Pairs"] - final["Concordant Pairs"]) ** 2)
            - (1 / n * (P - Q) ** 2)
        )
        / (d_var1 * d_var2)
    ) * 2
    ase0_tau_c = ase1_tau_c

    somers_sym = (P - Q) / (0.5 * (d_var2 + d_var1))
    somers_v1 = (P - Q) / d_var1
    somers_v2 = (P - Q) / d_var2

    _common_sq = np.sum(final["Nij"] * (final["Disconcordant Pairs"] - final["Concordant Pairs"]) ** 2) - (
        1 / n * (P - Q) ** 2
    )
    ase0_delta_sym = (4 / (d_var1 + d_var2)) * np.sqrt(_common_sq)
    ase0_delta_v1 = (2 / d_var1) * np.sqrt(_common_sq)
    ase0_delta_v2 = (2 / d_var2) * np.sqrt(_common_sq)
    ase1_delta_sym = (ase1_tau_b * 2 / (d_var1 + d_var2)) * np.sqrt(d_var1 * d_var2)
    ase1_delta_v1 = ase0_delta_v1
    ase1_delta_v2 = (2 / d_var2 ** 2) * np.sqrt(
        np.sum(
            final["Nij"]
            * (d_var2 * (final["Concordant Pairs"] - final["Disconcordant Pairs"]) - (P - Q) * (n - final["Nj"])) ** 2
        )
    )

    gamma = (P - Q) / (P + Q)
    ase1_gamma = (4 / (P + Q) ** 2) * np.sqrt(
        np.sum(
            final["Nij"]
            * np.float64(
                (final["Disconcordant Pairs"] * P - final["Concordant Pairs"] * Q) ** 2
            )
        )
    )
    ase0_gamma = (2 / (P + Q)) * np.sqrt(_common_sq)

    wilsons_e = (
        2
        * (
            np.sum(final["Concordant Pairs"] * final["Nij"] / 2)
            - np.sum(final["Disconcordant Pairs"] * final["Nij"] / 2)
        )
        / (n ** 2 - np.sum(np.sum(ct ** 2)))
    )
    ase_wilson_term1 = (
        4 * np.sum(final["Nij"] * (final["Concordant Pairs"] - final["Disconcordant Pairs"]) ** 2)
        - 4
        / n
        * (
            np.sum(final["Concordant Pairs"] * final["Nij"]) / 2
            - np.sum(final["Disconcordant Pairs"] * final["Nij"]) / 2
        )
        ** 2
    )
    ase_wilson_term2 = (n ** 2 - np.sum(final["Nij"] ** 2)) ** 2
    ase_wilson = np.sqrt(ase_wilson_term1 / ase_wilson_term2)

    z_tau_a = tau_a / ase0_tau_a
    z_tau_b = tau_b / ase0_tau_b
    z_tau_c = tau_c / ase0_tau_c
    z_somers_v1 = somers_v1 / ase0_delta_v1
    z_somers_v2 = somers_v2 / ase0_delta_v2
    z_somers_sym = somers_sym / ase0_delta_sym
    z_gamma = gamma / ase0_gamma
    z_wilson = wilsons_e / ase_wilson

    p_tau_a = norm.sf(z_tau_a)
    p_tau_b = norm.sf(z_tau_b)
    p_tau_c = norm.sf(z_tau_c)
    p_somers_v1 = norm.sf(z_somers_v1)
    p_somers_v2 = norm.sf(z_somers_v2)
    p_somers_sym = norm.sf(z_somers_sym)
    p_gamma = norm.sf(z_gamma)
    p_wilson = norm.sf(z_wilson)

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    ci_tau_a = [tau_a - zcrit * ase1_tau_a, tau_a + zcrit * ase1_tau_a]
    ci_tau_b = [tau_b - zcrit * ase1_tau_b, tau_b + zcrit * ase1_tau_b]
    ci_tau_c = [tau_c - zcrit * ase1_tau_c, tau_c + zcrit * ase1_tau_c]
    ci_somers_v1 = [somers_v1 - zcrit * ase1_delta_v1, somers_v1 + zcrit * ase1_delta_v1]
    ci_somers_v2 = [somers_v2 - zcrit * ase1_delta_v2, somers_v2 + zcrit * ase1_delta_v2]
    ci_somers_sym = [somers_sym - zcrit * ase1_delta_sym, somers_sym + zcrit * ase1_delta_sym]
    ci_gamma = [gamma - zcrit * ase1_gamma, gamma + zcrit * ase1_gamma]
    ci_wilson = [wilsons_e - zcrit * ase_wilson, wilsons_e + zcrit * ase_wilson]

    results = {
        "Sommer's Delta Symmetric": np.array([somers_sym, ase1_delta_sym, ase0_delta_sym]),
        "Sommer's Delta Variable 1": np.array([somers_v1, ase1_delta_v1, ase0_delta_v1]),
        "Sommer's Delta Variable 2": np.array([somers_v2, ase1_delta_v2, ase0_delta_v2]),
        "Kendall's Tau A": np.array([tau_a, ase1_tau_a, ase1_tau_a]),
        "Kendall's Tau B": np.array([tau_b, ase1_tau_b, ase0_tau_b]),
        "Stuart Tau C": np.array([tau_c, ase1_tau_c, ase0_tau_c]),
        "Gamma Correlation": np.array([gamma, ase1_gamma, ase0_gamma]),
        "Wilson's e": [wilsons_e, ase_wilson_term1, ase_wilson],
        "Weighted Tau": np.array(wtau),
        "Weighted Tau p-value": np.array(wtau_p),
        "Z-Statistic_tau_a": z_tau_a,
        "Z-Statistic_tau_b": z_tau_b,
        "Z-Statistic_tau_c": z_tau_c,
        "Z-Statistic_somers_delta_v1": z_somers_v1,
        "Z-Statistic_somers_delta_v2": z_somers_v2,
        "Z-Statistic_somers_delta_symmetric": z_somers_sym,
        "Z-Statistic_Gamma": z_gamma,
        "Z-Statistic_Wilson": z_wilson,
        "p-value_tau_a": p_tau_a,
        "p-value_tau_b": p_tau_b,
        "p-value_tau_c": p_tau_c,
        "p-value_somers_v1": p_somers_v1,
        "p-value_somers_v2": p_somers_v2,
        "p-value_somers_symmetric": p_somers_sym,
        "p-value_gamma": p_gamma,
        "p-value_wilson": p_wilson,
        "CI_tau_a": ci_tau_a,
        "CI_tau_b": ci_tau_b,
        "CI_tau_c": ci_tau_c,
        "CI_somers_v1": ci_somers_v1,
        "CI_somers_v2": ci_somers_v2,
        "CI_somers_symmetric": ci_somers_sym,
        "CI_gamma": ci_gamma,
        "CI_wilson": ci_wilson,
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


class OrdinalByOrdinal:
    """Ordinal by Ordinal correlation measures.

    Computes skipped correlation, Gaussian rank correlation, Gini's Gamma,
    Shepherd's Pi, Gamma-family measures (Tau-a/b/c, Somers' Delta, Gamma,
    Wilson's e), and Spearman correlation from raw data or a contingency
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
            Result dictionary containing all ordinal-by-ordinal correlation
            measures.
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
        ], dtype=float)
        var2 = np.array([
            i + 1
            for i in range(table.shape[0])
            for j in range(table.shape[1])
            for _ in range(table[i, j])
        ], dtype=float)

        return {
            "Contingency Table": table,
            "Skipped Correlation": _skipped_correlation(var1, var2, cl),
            "Gaussian Rank Correlation": _gaussian_rank_correlation(var1, var2, cl),
            "Ginni's Gamma": _ginis_gamma(var1, var2, cl),
            "Shepherd's Pi": _shepherd(var1, var2, n_boot, cl),
            "The Gamma Family Measures": _gamma_family_measures(var1, var2, cl),
            "Spearman Correlation": _spearman_correlation(var1, var2, cl),
        }

    @staticmethod
    def from_data(params: dict) -> dict:
        """Calculate measures from raw data vectors.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Variable 1"`` : array-like, first ordinal variable.
            - ``"Variable 2"`` : array-like, second ordinal variable.
            - ``"Confidence Level"`` : float, percentage (e.g. 95).
            - ``"Number Of Bootstraps Samples"`` : int.

        Returns
        -------
        dict
            Result dictionary containing all ordinal-by-ordinal correlation
            measures.
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
            "The Gamma Family Measures": _gamma_family_measures(var1, var2, cl),
            "Spearman Correlation": _spearman_correlation(var1, var2, cl),
        }
