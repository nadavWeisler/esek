"""Module for Nominal by Interval correlation measures."""

import math

import numpy as np
from scipy.stats import pearsonr, ncf, f, norm


def _non_central_ci_f(f_statistic, df1, df2, confidence_level):
    """Calculate non-central F confidence interval.

    Parameters
    ----------
    f_statistic : float
        Observed F statistic.
    df1 : int
        Numerator degrees of freedom.
    df2 : int
        Denominator degrees of freedom.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the non-central CI.
    """
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = 1 - upper_limit
    lower_ci_difference_value = 1

    def _lower_ci(f_statistic, df1, df2, upper_limit, lower_ci_difference_value):
        lower_bound = [0.001, f_statistic / 2, f_statistic]
        while ncf.cdf(f_statistic, df1, df2, lower_bound[0]) < upper_limit:
            return [0, f.cdf(f_statistic, df1, df2)] if f.cdf(f_statistic, df1, df2) < upper_limit else None
            lower_bound = [lower_bound[0] / 4, lower_bound[0], lower_bound[2]]
        while ncf.cdf(f_statistic, df1, df2, lower_bound[2]) > upper_limit:
            lower_bound = [lower_bound[0], lower_bound[2], lower_bound[2] + f_statistic]
        while lower_ci_difference_value > 0.0000001:
            lower_bound = (
                [lower_bound[0], (lower_bound[0] + lower_bound[1]) / 2, lower_bound[1]]
                if ncf.cdf(f_statistic, df1, df2, lower_bound[1]) < upper_limit
                else [lower_bound[1], (lower_bound[1] + lower_bound[2]) / 2, lower_bound[2]]
            )
            lower_ci_difference_value = abs(ncf.cdf(f_statistic, df1, df2, lower_bound[1]) - upper_limit)
        return [lower_bound[1]]

    def _upper_ci(f_statistic, df1, df2, lower_limit, lower_ci_difference_value):
        upper_bound = [f_statistic, 2 * f_statistic, 3 * f_statistic]
        while ncf.cdf(f_statistic, df1, df2, upper_bound[0]) < lower_limit:
            upper_bound = [upper_bound[0] / 4, upper_bound[0], upper_bound[2]]
        while ncf.cdf(f_statistic, df1, df2, upper_bound[2]) > lower_limit:
            upper_bound = [upper_bound[0], upper_bound[2], upper_bound[2] + f_statistic]
        while lower_ci_difference_value > 0.00001:
            upper_bound = (
                [upper_bound[0], (upper_bound[0] + upper_bound[1]) / 2, upper_bound[1]]
                if ncf.cdf(f_statistic, df1, df2, upper_bound[1]) < lower_limit
                else [upper_bound[1], (upper_bound[1] + upper_bound[2]) / 2, upper_bound[2]]
            )
            lower_ci_difference_value = abs(ncf.cdf(f_statistic, df1, df2, upper_bound[1]) - lower_limit)
        return [upper_bound[1]]

    lower_ci_final = _lower_ci(f_statistic, df1, df2, upper_limit, lower_ci_difference_value)[0]
    upper_ci_final = _upper_ci(f_statistic, df1, df2, lower_limit, lower_ci_difference_value)[0]
    return lower_ci_final, upper_ci_final


def _point_biserial_correlation(x, y, confidence_level):
    """Compute point-biserial correlation and confidence intervals.

    Parameters
    ----------
    x : array-like
        Binary nominal variable (0/1 coded).
    y : array-like
        Continuous interval variable.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    pearson_result = pearsonr(x, y)
    rpb, pvalue = pearson_result

    x_sorted = x[np.argsort(x)]
    y_sorted = y[np.argsort(y)]
    ss_total = np.var(y) * len(x)
    ss_between_max = np.sum([
        len(y_sorted[x_sorted == v]) * (np.mean(y_sorted[x_sorted == v]) - np.mean(y_sorted)) ** 2
        for v in np.unique(x_sorted)
    ])
    pearson_max = np.sqrt(ss_between_max / ss_total)
    rpb_max_corrected = float(np.clip(rpb / pearson_max, -0.9999999, 0.9999999))
    rpb_approximated = float(np.clip(rpb + (rpb * (1 - rpb ** 2)) / (2 * (len(x) - 3)), -0.9999999, 0.9999999))

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_ci, upper_ci = pearson_result.confidence_interval(confidence_level)
    se = 1 / np.sqrt(len(x) - 3)

    lower_max = math.tanh(math.atanh(rpb_max_corrected) - zcrit * se)
    upper_max = math.tanh(math.atanh(rpb_max_corrected) + zcrit * se)
    lower_approx = math.tanh(math.atanh(rpb_approximated) - zcrit * se)
    upper_approx = math.tanh(math.atanh(rpb_approximated) + zcrit * se)

    results = {
        "Point Biserial Correlation": rpb,
        "p-value": pvalue,
        "Confidence Intervals Point Biserial Correlation (Fisher)": f"({round(lower_ci, 4)}, {round(upper_ci, 4)})",
        "Approximated point biserial correlation (Hedges & Olkin, 1985)": rpb_approximated,
        "Fisher Transformed Confidence Intervals Approximated Point Biserial Correlation)": (
            f"({round(lower_approx, 4)}, {round(upper_approx, 4)})"
        ),
        "Max Corrected point biserial correlation (Hedges & Olkin, 1985)": rpb_max_corrected,
        "Fisher Transformed Confidence Intervals Max Corrected Point Biserial Correlation)": (
            f"({round(lower_max, 4)}, {round(upper_max, 4)})"
        ),
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


def _eta_correlation_ratio(x, y, confidence_level):
    """Compute eta correlation ratio and confidence intervals.

    Parameters
    ----------
    x : array-like
        Nominal (grouping) variable.
    y : array-like
        Continuous interval variable.
    confidence_level : float
        Confidence level (0–1).

    Returns
    -------
    str
        Formatted results string.
    """
    ss_between_x = np.sum([
        len(y[x == v]) * (np.mean(y[x == v]) - np.mean(y)) ** 2
        for v in np.unique(x)
    ])
    ss_total_x = np.sum((y - np.mean(y)) ** 2)
    eta_x = np.sqrt(ss_between_x / ss_total_x)

    ss_between_y = np.sum([
        len(x[y == v]) * (np.mean(x[y == v]) - np.mean(x)) ** 2
        for v in np.unique(y)
    ])
    ss_total_y = np.sum((x - np.mean(x)) ** 2)
    eta_y = np.sqrt(ss_between_y / ss_total_y)

    x_sorted = x[np.argsort(x)]
    y_sorted = y[np.argsort(y)]
    ss_between_x_max = np.sum([
        len(y_sorted[x_sorted == v]) * (np.mean(y_sorted[x_sorted == v]) - np.mean(y_sorted)) ** 2
        for v in np.unique(x_sorted)
    ])
    eta_max_x = np.sqrt(ss_between_x_max / ss_total_x)

    ss_between_y_max = np.sum([
        len(x_sorted[y_sorted == v]) * (np.mean(x_sorted[y_sorted == v]) - np.mean(x_sorted)) ** 2
        for v in np.unique(y_sorted)
    ])
    eta_max_y = np.sqrt(ss_between_y_max / ss_total_y)

    eta_x_corrected = eta_x / eta_max_x
    eta_y_corrected = eta_y / eta_max_y

    n_groups = len(np.unique(x))
    n = len(x)
    df1 = n_groups - 1
    df2 = n - n_groups

    f_x = (-eta_x ** 2 * df2) / (df1 * (eta_x ** 2 - 1))
    f_y = (-eta_y ** 2 * df2) / (df1 * (eta_y ** 2 - 1))
    f_cx = (-eta_x_corrected ** 2 * df2) / (df1 * (eta_x_corrected ** 2 - 1))
    f_cy = (-eta_y_corrected ** 2 * df2) / (df1 * (eta_y_corrected ** 2 - 1))

    lo_x, hi_x = _non_central_ci_f(f_x, df1, df2, confidence_level)
    lo_y, hi_y = _non_central_ci_f(f_y, df1, df2, confidence_level)
    lo_cx, hi_cx = _non_central_ci_f(f_cx, df1, df2, confidence_level)
    lo_cy, hi_cy = _non_central_ci_f(f_cy, df1, df2, confidence_level)

    results = {
        "Eta (Variable X is Independent)": eta_x,
        "Eta (Variable X is Independent) - Confidence Intervals": [
            np.sqrt(lo_x / (lo_x + df2)),
            np.sqrt(hi_x / (hi_x + df2)),
        ],
        "Eta (Variable Y is Independent)": eta_y,
        "Eta (Variable Y is Independent) - Confidence Intervals": [
            np.sqrt(lo_y / (lo_y + df2)),
            np.sqrt(hi_y / (hi_y + df2)),
        ],
        "Attenuated Correct Eta (Variable X is Independent)": eta_x_corrected,
        "Attenuated Correct Eta (Variable X is Independent) - Confidence Intervals": [
            np.sqrt(lo_cx / (lo_cx + df2)),
            np.sqrt(hi_cx / (hi_cx + df2)),
        ],
        "Attenuated Correct Eta (Variable Y is Independent)": eta_y_corrected,
        "Attenuated Correct Eta (Variable Y is Independent) - Confidence Intervals": [
            np.sqrt(lo_cy / (lo_cy + df2)),
            np.sqrt(hi_cy / (hi_cy + df2)),
        ],
    }
    return "\n".join([f"{k}: {v}" for k, v in results.items()])


class NominalByInterval:
    """Nominal by Interval correlation measures.

    Computes point-biserial correlation and eta correlation ratio
    from raw data or a contingency table.
    """

    @staticmethod
    def from_data(params: dict) -> dict:
        """Calculate measures from raw data vectors.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Nominal"`` : array-like, nominal (grouping) variable.
            - ``"Interval"`` : array-like, continuous interval variable.
            - ``"Confidence Level"`` : float, confidence level as a percentage
              (e.g. 95 for 95%).

        Returns
        -------
        dict
            Dictionary with eta output and point-biserial correlation output.
        """
        nominal = np.array(params["Nominal"])
        interval = np.array(params["Interval"])
        cl_pct = params["Confidence Level"]
        cl = cl_pct / 100

        # encode non-numeric nominal values
        nominal = np.array([
            ({v: i for i, v in enumerate(np.unique(nominal))})[v]
            for v in nominal
        ])

        rpb_out = (
            _point_biserial_correlation(nominal, interval, cl)
            if len(np.unique(nominal)) == 2
            else "Point Biserial Correlation only relevant when the Nominal Variable has 2 levels"
        )
        eta_out = _eta_correlation_ratio(nominal, interval, cl)

        return {
            "eta_output": eta_out,
            "_______________________": "",
            "Point Biserial Correlation": rpb_out,
        }

    @staticmethod
    def from_contingency_table(params: dict) -> dict:
        """Calculate measures from a contingency table.

        Parameters
        ----------
        params : dict
            Keys:
            - ``"Table"`` : 2-D numpy array, contingency table where rows are
              nominal categories and columns are interval categories.
            - ``"Confidence Level"`` : float, confidence level as a percentage.

        Returns
        -------
        dict
            Dictionary with eta output and point-biserial correlation output.
        """
        table = params["Table"]
        cl_pct = params["Confidence Level"]
        cl = cl_pct / 100

        interval = np.array([
            j + 1
            for i in range(table.shape[0])
            for j in range(table.shape[1])
            for _ in range(table[i, j])
        ])
        nominal = np.array([
            i + 1
            for i in range(table.shape[0])
            for j in range(table.shape[1])
            for _ in range(table[i, j])
        ])

        rpb_out = (
            _point_biserial_correlation(np.array(nominal), np.array(interval), cl)
            if len(np.unique(nominal)) == 2
            else "Point Biserial Correlation only relevant when the Nominal Variable has 2 levels"
        )
        eta_out = _eta_correlation_ratio(np.array(nominal), np.array(interval), cl)

        return {
            "eta_output": eta_out,
            "_______________________": "",
            "Point Biserial Correlation": rpb_out,
        }
