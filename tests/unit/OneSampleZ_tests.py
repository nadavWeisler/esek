"""
Unit tests for the One_Sample_ZTests module.

This module contains unit tests for the One_Sample_ZTests module, which includes
functions for performing one-sample Z-tests and calculating various statistics
such as Cohen's d, Z-score, p-value, confidence intervals, and standard errors.

The tests cover the following functions:
- one_sample_from_z_score
- one_sample_from_parameters
- one_sample_from_data
"""

import unittest
from OneSampleZ import OneSampleZ

class TestOneSampleZTests(unittest.TestCase):
    """
    Unit tests for the One_Sample_ZTests module.

    This class contains tests for the following functions:
    - one_sample_from_z_score
    - one_sample_from_parameters
    - one_sample_from_data
    """
    def test_one_sample_from_z_score(self):
        """
        Test the one_sample_from_z_score function from One_Sample_ZTests.

        This test checks the calculation of Cohen's d, Z-score, p-value,
        Cohen's d confidence interval, and the standard error of the effect size
        using a given Z-score, sample size, and confidence level.
        """
        params = {"Z-score": 0.2667, "Sample Size": 25, "Confidence Level": 95}
        results = OneSampleZ.one_sample_from_z_score(params)

        self.assertAlmostEqual(results["Cohen's d"], 0.0533, places=4)
        self.assertAlmostEqual(results["Z-score"], 0.2667, places=4)
        self.assertAlmostEqual(results["p-value"], 0.7897, places=4)
        self.assertAlmostEqual(results["Cohen's d CI Lower"], -0.3389, places=4)
        self.assertAlmostEqual(results["Cohen's d CI Upper"], 0.4456, places=4)
        self.assertAlmostEqual(
            results["Standard Error of the Effect Size"], 0.2001, places=4
        )

    def test_one_sample_from_parameters(self):
        """
        Test the one_sample_from_parameters function from One_Sample_ZTests.

        This test checks the calculation of Cohen's d, Z-score, p-value,
        standard error of the mean, Cohen's d confidence interval, and the
        standard error of the effect size using population mean, population
        standard deviation, sample's mean, sample size, and confidence level.
        """
        params = {
            "Population Mean": 50,
            "Population Standard Deviation": 15,
            "Sample's Mean": 49.2,
            "Sample Size": 25,
            "Confidence Level": 95,
        }
        results = OneSampleZ.one_sample_from_parameters(params)

        self.assertAlmostEqual(results["Cohen's d"], 0.0533, places=4)
        self.assertAlmostEqual(results["Z-score"], 0.2667, places=4)
        self.assertAlmostEqual(results["p-value"], 0.7897, places=4)
        self.assertAlmostEqual(results["Standart Error of the Mean"], 3.0, places=4)
        self.assertAlmostEqual(results["Cohen's d CI Lower"], -0.3389, places=4)
        self.assertAlmostEqual(results["Cohen's d CI Upper"], 0.4456, places=4)
        self.assertAlmostEqual(
            results["Standrd Error of the Effect Size"], 0.2001, places=4
        )

    def test_one_sample_from_data(self):
        """
        Test the one_sample_from_data function from One_Sample_ZTests.

        This test checks the calculation of Cohen's d, Z-score, p-value,
        standard error of the mean, Cohen's d confidence interval, standard
        error of the effect size, sample's mean, sample's standard deviation,
        difference between means, and sample size using a data list, population
        mean, population standard deviation, and confidence level.
        """
        data = [
            45,
            48,
            51,
            49,
            52,
            47,
            50,
            53,
            48,
            50,
            46,
            51,
            48,
            50,
            49,
            52,
            47,
            50,
            45,
            53,
            49,
            51,
            46,
            52,
            48,
        ]
        params = {
            "column_1": data,
            "Population Mean": 50,
            "Popoulation Standard Deviation": 15,
            "Confidence Level": 95.5,
        }
        results = OneSampleZ.one_sample_from_data(params)

        self.assertAlmostEqual(results["Cohen's d"], 0.0533, places=4)
        self.assertAlmostEqual(results["Z-score"], 0.2667, places=4)
        self.assertAlmostEqual(results["p-value"], 0.7897, places=4)
        self.assertAlmostEqual(results["Standart Error of the Mean"], 3.0, places=4)
        self.assertAlmostEqual(results["Cohen's d CI Lower"], -0.3479, places=4)
        self.assertAlmostEqual(results["Cohen's d CI Upper"], 0.4545, places=4)
        self.assertAlmostEqual(
            results["Standrd Error of the Effect Size"], 0.2001, places=4
        )
        self.assertAlmostEqual(results["Sample's Mean"], 49.2, places=4)
        self.assertAlmostEqual(results["Sample's Standard Deviation"], 2.3805, places=4)
        self.assertAlmostEqual(results["Difference Between Means"], 0.8, places=4)
        self.assertEqual(results["Sample Size"], 25)


if __name__ == "__main__":
    unittest.main()
