import unittest
import numpy as np
from One_Sample_CLES import One_Sample_ttest

class TestOneSampleCLES(unittest.TestCase):
    def test_one_sample_from_t_score(self):
        params = {
            "t score": 1.1175,
            "Sample Size": 25,
            "Confidence Level": 95
        }
        results = One_Sample_ttest.one_sample_from_t_score(params)
        
        # Test confidence intervals for CLd
        self.assertAlmostEqual(results["Lower Central CI's CLd"], 42.5731, places=4)
        self.assertAlmostEqual(results["Upper Central CI's CLd"], 74.0041, places=4)
        self.assertAlmostEqual(results["Lower Pivotal CI's CLd"], 43.0232, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's CLd"], 73.1787, places=4)
        self.assertAlmostEqual(results["Lower Non-Central CI's CLd"], 43.3110, places=4)
        self.assertAlmostEqual(results["Upper Non-Central CI's CLd"], 74.8115, places=4)
        
        # Test confidence intervals for CLg
        self.assertAlmostEqual(results["Lower Central CI's CLg"], 42.3048, places=4)
        self.assertAlmostEqual(results["Upper Central CI's CLg"], 73.7575, places=4)
        self.assertAlmostEqual(results["Lower Pivotal CI's CLg"], 43.2418, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's CLg"], 72.5303, places=4)
        self.assertAlmostEqual(results["Lower Non-Central CI's CLg"], 43.0142, places=4)
        self.assertAlmostEqual(results["Upper Non-Central CI's CLg"], 74.5472, places=4)

    def test_one_sample_from_params(self):
        params = {
            "Population Mean": 29,
            "Mean Sample": 32.76,
            "Standard Deviation Sample": 16.483,
            "Sample Size": 25,
            "Confidence Level": 95
        }
        results = One_Sample_ttest.one_sample_from_params(params)
        
        # Test key metrics
        self.assertAlmostEqual(results["Mcgraw & Wong, Common Language Effect Size (CLd)"], 59.0221, places=4)
        self.assertAlmostEqual(results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLg)"], 58.7414, places=4)
        self.assertAlmostEqual(results["t-score"], 1.1175, places=4)
        self.assertEqual(results["degrees of freedom"], 24)
        self.assertAlmostEqual(results["p-value"], 0.2748, places=4)
        
        # Test confidence intervals
        self.assertAlmostEqual(results["Lower Pivotal CI's CLd"], 43.0231, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's CLd"], 73.1791, places=4)
        self.assertAlmostEqual(results["Lower Pivotal CI's CLg"], 43.2416, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's CLg"], 72.5308, places=4)

    def test_one_sample_from_data(self):
        data = [21, 11, 51, 49, 52, 47, 12, 53, 17, 50, 14, 33, 11, 50, 49, 
                52, 47, 22, 25, 30, 19, 51, 16, 18, 19]
        params = {
            "column 1": data,
            "Population's Mean": 29,
            "Confidence Level": 95
        }
        results = One_Sample_ttest.one_sample_from_data(params)
        
        # Test key metrics
        self.assertAlmostEqual(results["Mcgraw & Wong, Common Language Effect Size (CLd)"], 59.0221, places=4)
        self.assertAlmostEqual(results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLg)"], 58.7414, places=4)
        self.assertAlmostEqual(results["t-score"], 1.1175, places=4)
        self.assertEqual(results["degrees of freedom"], 24)
        self.assertAlmostEqual(results["p-value"], 0.2748, places=4)
        
        # Test confidence intervals
        self.assertAlmostEqual(results["Lower Pivotal CI's CLd"], 43.0231, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's CLd"], 73.1791, places=4)
        self.assertAlmostEqual(results["Lower Pivotal CI's CLg"], 43.2416, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's CLg"], 72.5307, places=4)

    def test_robust_one_sample(self):
        data = [21, 11, 51, 49, 52, 47, 12, 53, 17, 50, 14, 33, 11, 50, 49, 
                52, 47, 22, 25, 30, 19, 51, 16, 18, 19]
        params = {
            "column 1": data,
            "Population Mean": 29,
            "Confidence Level": 95,
            "Number of Bootstrap Samples": 95,
            "Trimming Level": 0.2
        }
        results = One_Sample_ttest.Robust_One_Sample(params)
        
        # Test key metrics
        self.assertAlmostEqual(results["Robust Effect Size AKP"], -1.0654, places=4)
        self.assertAlmostEqual(results["Lower Confidence Interval Robust AKP"], -1.5367, places=4)
        self.assertAlmostEqual(results["Upper Confidence Interval Robust AKP"], -0.6932, places=4)
        self.assertAlmostEqual(results["Trimmed Mean 1"], 33.0667, places=4)
        self.assertAlmostEqual(results["Winsorized Standard Deviation 1"], 15.023, places=4)
        self.assertAlmostEqual(results["Yuen's T statistic"], 0.8121, places=4)
        self.assertEqual(results["Degrees of Freedom"], 14.0)
        self.assertAlmostEqual(results["p-value"], 0.4303, places=4)
        self.assertAlmostEqual(results["Difference Between Means"], 4.0667, places=4)
        self.assertAlmostEqual(results["Standard Error"], 5.0077, places=4)

if __name__ == '__main__':
    unittest.main() 