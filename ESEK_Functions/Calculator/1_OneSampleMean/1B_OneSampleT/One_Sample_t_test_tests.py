import unittest
import numpy as np
from One_Sample_t_test import One_Sample_ttest

class TestOneSampleTTest(unittest.TestCase):
    def test_one_sample_from_t_score(self):
        params = {
            "t score": 1.1175,
            "Sample Size": 25,
            "Confidence Level": 95.5
        }
        results = One_Sample_ttest.one_sample_from_t_score(params)
        
        # Test key metrics
        self.assertAlmostEqual(results["Cohen's d"], 0.2281, places=4)
        self.assertAlmostEqual(results["Hedges' g"], 0.2209, places=4)
        self.assertAlmostEqual(results["t score"], 1.1175, places=4)
        self.assertEqual(results["Degrees of Freedom"], 24)
        self.assertAlmostEqual(results["p-value"], 0.2748, places=4)
        
        # Test standard errors
        self.assertAlmostEqual(results["Standard Error of Cohen's d (True)"], 0.0449, places=4)
        self.assertAlmostEqual(results["Standard Error of Hedges' g (True)"], 0.0448, places=4)
        
        # Test confidence intervals
        self.assertAlmostEqual(results["Lower Pivotal CI's Cohen's d"], -0.1848, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's Cohen's d"], 0.6273, places=4)
        self.assertAlmostEqual(results["Lower Pivotal CI's Hedges' g"], -0.1789, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's Hedges' g"], 0.6074, places=4)

    def test_one_sample_from_params(self):
        params = {
            "Population Mean": 29,
            "Mean Sample": 32.76,
            "Standard Deviation Sample": 16.483,
            "Sample Size": 2500,
            "Confidence Level": 95.5
        }
        results = One_Sample_ttest.one_sample_from_params(params)
        
        # Test key metrics
        self.assertAlmostEqual(results["Cohen's d"], 0.2281, places=4)
        self.assertAlmostEqual(results["Hedges' g"], 0.2280, places=4)
        self.assertAlmostEqual(results["t score"], 11.4034, places=4)
        self.assertEqual(results["Degrees of Freedom"], 2499)
        self.assertAlmostEqual(results["p-value"], 0.0, places=4)
        
        # Test standard errors and standardizers
        self.assertAlmostEqual(results["Standard Error of the Mean"], 0.3297, places=4)
        self.assertAlmostEqual(results["Standardizer Cohen's d (Sample's Standard Deviation)"], 16.483, places=3)
        
        # Test confidence intervals
        self.assertAlmostEqual(results["Lower Pivotal CI's Cohen's d"], 0.1874, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's Cohen's d"], 0.2687, places=4)

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
        self.assertAlmostEqual(results["Cohen's d"], 0.2281, places=4)
        self.assertAlmostEqual(results["Hedges' g"], 0.2209, places=4)
        self.assertAlmostEqual(results["t_score"], 1.1175, places=4)
        self.assertEqual(results["Degrees of Freedom"], 24)
        self.assertAlmostEqual(results["p-value"], 0.2748, places=4)
        
        # Test sample statistics
        self.assertAlmostEqual(results["Sample's Mean"], 32.76, places=2)
        self.assertAlmostEqual(results["Sample's Standard Deviation"], 16.483, places=3)
        self.assertAlmostEqual(results["Means Difference"], 3.76, places=2)
        self.assertEqual(results["Sample Size"], 25)
        
        # Test confidence intervals
        self.assertAlmostEqual(results["Lower Pivotal CI's Cohen's d"], -0.1758, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's Cohen's d"], 0.6182, places=4)
        self.assertAlmostEqual(results["Lower Pivotal CI's Hedges' g"], -0.1702, places=4)
        self.assertAlmostEqual(results["Upper Pivotal CI's Hedges' g"], 0.5987, places=4)

if __name__ == '__main__':
    unittest.main() 