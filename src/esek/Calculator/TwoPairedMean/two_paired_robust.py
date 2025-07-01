"""
This module implements robust statistical tests for two paired samples.
It includes methods for calculating robust effect sizes, confidence intervals,
and inferential statistics using Yuen's t-test and winsorized Pearson correlation.
It also provides a data class to store the results of these tests.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from scipy.stats import t, norm, trim_mean
from ...utils import interfaces
from ...utils import res


@dataclass
class TwoPairedRobustResults:
    """
    A data class to store the results of two paired robust tests.
        It includes samples, inferential statistics, robust effect sizes,
    """

    sample1: Optional[res.Sample] = None
    sample2: Optional[res.Sample] = None
    inferential: Optional[res.InferentialStatistics] = None
    robust_akp: Optional[res.RobustAKP] = None
    robust_explanatory: Optional[res.RobustExplanatory] = None
    yuen_robust_t: Optional[float] = None
    winsorized_pearson_correlation: Optional[float] = None
    winsorized_pearson_correlation_p_value: Optional[float] = None


class TwoPairedRobustTests(interfaces.AbstractTest):
    """
    A class to perform robust statistical tests for two paired samples.
    It includes methods for calculating robust effect sizes, confidence intervals,
    and inferential statistics using Yuen's t-test and winsorized Pearson correlation.

        Methods:
                - from_score: Not implemented.
                - from_parameters: Not implemented.
                - from_data: Calculates robust statistics from two paired samples.
                - density: Calculates the density function for a given input.
                - area_under_function: Computes the area under a given function between two points.
                - winsorized_variance: Computes the winsorized variance of a sample.
                - winsorized_correlation: Computes the winsorized Pearson correlation between two samples.
    """

    @staticmethod
    def from_score() -> TwoPairedRobustResults:
        """
        A static method to create results from a score.
        This method is not implemented and will raise a NotImplementedError.
        Raises:
                NotImplementedError: This method is not implemented for TwoPairedRobustResults.

        """
        raise NotImplementedError(
            "from_score method is not implemented for TwoPairedRobust."
        )

    @staticmethod
    def from_parameters() -> TwoPairedRobustResults:
        """
        A static method to create results from parameters.
            This method is not implemented and will raise a NotImplementedError.
        Raises:
                    NotImplementedError: This method is not implemented for TwoPairedRobustResults.
        """
        raise NotImplementedError(
            "from_parameters method is not implemented for TwoPairedRobust."
        )

    def from_data(
        self,
        columns: list,
        reps: int,
        confidence_level: float,
        trimming_level: float = 0.2,
        population_difference: float = 0.2,
    ) -> TwoPairedRobustResults:
        """
        A method to calculate robust statistics from two paired samples.
        """

        column_1 = columns[0]
        column_2 = columns[1]

        difference = column_1 - column_2
        sample_size = len(column_1)
        correction = np.sqrt(
            float(
                self.area_under_function(
                    self.density,
                    float(norm.ppf(trimming_level)),
                    float(norm.ppf(1 - trimming_level)),
                )
            )
            + 2 * (norm.ppf(trimming_level) ** 2) * trimming_level
        )
        trimmed_mean_1 = trim_mean(column_1, trimming_level)
        trimmed_mean_2 = trim_mean(column_2, trimming_level)
        winsorized_standard_deviation_1 = np.sqrt(self.winsorized_variance(column_1))
        winsorized_standard_deviation_2 = np.sqrt(self.winsorized_variance(column_2))
        winsorized = self.winsorized_correlation(column_1, column_2, trimming_level)
        winsorized_correlation = winsorized["cor"]
        winsorized_correlation_p_value = winsorized["p.value"]

        standardizer = np.sqrt(self.winsorized_variance(difference, trimming_level))
        trimmed_mean = trim_mean(difference, trimming_level)
        akp_effect_size = (
            correction * (trimmed_mean - population_difference) / standardizer
        )

        bootstrap_samples_difference = []
        for _ in range(reps):
            difference_bootstrap = np.random.choice(
                difference, len(difference), replace=True
            )
            bootstrap_samples_difference.append(difference_bootstrap)

        trimmed_means_of_bootstrap = trim_mean(
            bootstrap_samples_difference, trimming_level, axis=1
        )
        standardizer_of_bootstrap = np.sqrt(
            [
                self.winsorized_variance(array, trimming_level)
                for array in bootstrap_samples_difference
            ]
        )
        akp_effect_size_bootstrap = (
            correction
            * (trimmed_means_of_bootstrap - population_difference)
            / standardizer_of_bootstrap
        )
        lower_ci_akp_boot = np.percentile(
            akp_effect_size_bootstrap,
            ((1 - confidence_level) - ((1 - confidence_level) / 2)) * 100,
        )
        upper_ci_akp_boot = np.percentile(
            akp_effect_size_bootstrap,
            ((confidence_level) + ((1 - confidence_level) / 2)) * 100,
        )

        sort_values = np.concatenate((column_1, column_2))
        variance_between_trimmed_means = (
            np.std(np.array([trimmed_mean_1, trimmed_mean_2]), ddof=1)
        ) ** 2
        winsorized_variance_value = self.winsorized_variance(
            sort_values, trimming_level
        )
        explained_variance = variance_between_trimmed_means / (
            winsorized_variance_value / correction**2
        )
        explanatory_power_effect_size = min(float(np.sqrt(explained_variance)), 1)

        bootstrap_samples_x = []
        bootstrap_samples_y = []
        for _ in range(reps):
            sample_1_bootstrap = np.random.choice(column_1, len(column_1), replace=True)
            sample_2_bootstrap = np.random.choice(column_2, len(column_2), replace=True)
            bootstrap_samples_x.append(sample_1_bootstrap)
            bootstrap_samples_y.append(sample_2_bootstrap)

        concatenated_samples = [
            np.concatenate((x, y))
            for x, y in zip(bootstrap_samples_x, bootstrap_samples_y)
        ]
        trimmed_means_of_bootstrap_sample_1 = np.array(
            (trim_mean(bootstrap_samples_x, trimming_level, axis=1))
        )
        trimmed_means_of_bootstrap_sample_2 = np.array(
            (trim_mean(bootstrap_samples_y, trimming_level, axis=1))
        )
        variance_between_trimmed_means_bootstrap = [
            (np.std(np.array([x, y]), ddof=1)) ** 2
            for x, y in zip(
                trimmed_means_of_bootstrap_sample_1, trimmed_means_of_bootstrap_sample_2
            )
        ]
        winsorized_variances_bootstrap = [
            self.winsorized_variance(arr, trimming_level)
            for arr in concatenated_samples
        ]
        explained_variance_bootstrapping = np.array(
            variance_between_trimmed_means_bootstrap
            / (winsorized_variances_bootstrap / correction**2)
        )
        explanatory_power_effect_size_bootstrap = [
            array**0.5 for array in explained_variance_bootstrapping
        ]
        lower_ci_e_pow_boot = np.percentile(
            explanatory_power_effect_size_bootstrap,
            ((1 - confidence_level) - ((1 - confidence_level) / 2)) * 100,
        )
        upper_ci_e_pow_boot = np.percentile(
            explanatory_power_effect_size_bootstrap,
            ((confidence_level) + ((1 - confidence_level) / 2)) * 100,
        )

        q1 = (len(column_1) - 1) * winsorized_standard_deviation_1**2
        q2 = (len(column_2) - 1) * winsorized_standard_deviation_2**2
        q3 = (len(column_1) - 1) * winsorized["cov"]
        non_winsorized_sample_size = len(column_1) - 2 * np.floor(
            trimming_level * len(column_1)
        )
        df = non_winsorized_sample_size - 1
        yuen_standard_error = np.sqrt(
            (q1 + q2 - 2 * q3)
            / (non_winsorized_sample_size * (non_winsorized_sample_size - 1))
        )
        difference_trimmed_means = trimmed_mean_1 - trimmed_mean_2
        yuen_statistic = difference_trimmed_means / yuen_standard_error
        yuen_p_value = 2 * (1 - t.cdf(np.abs(yuen_statistic), df))

        sample1 = res.Sample(
            mean=round(trimmed_mean_1, 4),
            standard_deviation=round(winsorized_standard_deviation_1, 4),
            size=round(sample_size, 4),
        )
        sample2 = res.Sample(
            mean=round(trimmed_mean_2, 4),
            standard_deviation=round(winsorized_standard_deviation_2, 4),
            size=round(sample_size, 4),
        )

        inferential = res.InferentialStatistics(
            p_value=float(np.around(yuen_p_value, 4)),
            score=round(yuen_statistic, 4),
        )
        inferential.standard_error = round(yuen_standard_error, 4)
        inferential.degrees_of_freedom = round(df, 4)
        inferential.means_difference = round(difference_trimmed_means, 4)

        robust_akp = res.RobustAKP(
            value=round(akp_effect_size, 4),
            ci_lower=round(lower_ci_akp_boot, 4),
            ci_upper=round(upper_ci_akp_boot, 4),
            standard_error=round(yuen_standard_error, 4),
        )
        robust_explanatory = res.RobustExplanatory(
            value=round(explanatory_power_effect_size, 4),
            ci_lower=round(lower_ci_e_pow_boot, 4),
            ci_upper=round(min(float(upper_ci_e_pow_boot), 1.0), 4),
            standard_error=round(yuen_standard_error, 4),
        )

        results = TwoPairedRobustResults()
        results.sample1 = sample1
        results.sample2 = sample2
        results.inferential = inferential
        results.robust_akp = robust_akp
        results.robust_explanatory = robust_explanatory
        results.yuen_robust_t = round(yuen_statistic, 4)
        results.winsorized_pearson_correlation = round(winsorized_correlation, 4)
        results.winsorized_pearson_correlation_p_value = round(
            winsorized_correlation_p_value, 4
        )

        return results

    def density(self, x):
        x = np.array(x)
        return x**2 * norm.pdf(x)

    def area_under_function(
        self,
        f: Callable,
        a,
        b,
        *args,
        function_a=None,
        function_b=None,
        limit=10,
        eps=1e-5,
    ):
        if function_a is None:
            function_a = f(a, *args)
        if function_b is None:
            function_b = f(b, *args)
        midpoint = (a + b) / 2
        f_midpoint = f(midpoint, *args)
        area_trapezoidal = ((function_a + function_b) * (b - a)) / 2
        area_simpson = ((function_a + 4 * f_midpoint + function_b) * (b - a)) / 6
        if abs(area_trapezoidal - area_simpson) < eps or limit == 0:
            return area_simpson
        return self.area_under_function(
            f,
            a,
            midpoint,
            *args,
            function_a=function_a,
            function_b=f_midpoint,
            limit=limit - 1,
            eps=eps,
        ) + self.area_under_function(
            f,
            midpoint,
            b,
            *args,
            function_a=f_midpoint,
            function_b=function_b,
            limit=limit - 1,
            eps=eps,
        )

    def winsorized_variance(self, x, trimming_level=0.2):
        y = np.sort(x)
        n = len(x)
        ibot = int(np.floor(trimming_level * n)) + 1
        itop = n - ibot + 1
        xbot = y[ibot - 1]
        xtop = y[itop - 1]
        y = np.where(y <= xbot, xbot, y)
        y = np.where(y >= xtop, xtop, y)
        winvar = np.std(y, ddof=1) ** 2
        return winvar

    def winsorized_correlation(self, x, y, trimming_level=0.2):
        sample_size = len(x)
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        trimming_size = int(np.floor(trimming_level * sample_size)) + 1
        x_lower = x_sorted[trimming_size - 1]
        x_upper = x_sorted[sample_size - trimming_size]
        y_lower = y_sorted[trimming_size - 1]
        y_upper = y_sorted[sample_size - trimming_size]
        x_winsorized = np.clip(x, x_lower, x_upper)
        y_winsorized = np.clip(y, y_lower, y_upper)
        winsorized_correlation = np.corrcoef(x_winsorized, y_winsorized)[0, 1]
        winsorized_covariance = np.cov(x_winsorized, y_winsorized)[0, 1]
        test_statistic = winsorized_correlation * np.sqrt(
            (sample_size - 2) / (1 - winsorized_correlation**2)
        )
        number_of_trimmed_values = int(np.floor(trimming_level * sample_size))
        p_value = 2 * (
            1
            - t.cdf(
                np.abs(test_statistic), sample_size - 2 * number_of_trimmed_values - 2
            )
        )
        return {
            "cor": winsorized_correlation,
            "cov": winsorized_covariance,
            "p.value": p_value,
            "n": sample_size,
            "test_statistic": test_statistic,
        }
