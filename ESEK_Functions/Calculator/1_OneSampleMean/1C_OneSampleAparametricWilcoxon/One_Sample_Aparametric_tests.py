import unittest
import numpy as np
from One_Sample_Aparametric import Aparametric_One_Sample

class TestOneSampleAparametric(unittest.TestCase):
    def test_aparametric_effect_size_onesample(self):
        data = [11, 0, 39, 35, 31, 30, 0, 34, -2, 28, -8, 1, 0, 17, 2, 5, 
                -2, -27, -1, -20, -2, 0, -36, -34, -34]
        params = {
            "Column 1": np.array(data),
            "Confidence Level": 95,
            "Population's Value": 0
        }
        results = Aparametric_One_Sample.Apermetric_Effect_Size_onesample(params)
        
        # Test basic statistics
        self.assertEqual(results["Sample"], 175.0)
        self.assertEqual(results["Sample Median"], 0.0)
        self.assertEqual(results["Median of the Differnece"], 0.0)
        self.assertEqual(results["Median of Absoult Deviation"], 11.0)
        self.assertAlmostEqual(results["Sample Mean"], 2.68, places=2)
        self.assertAlmostEqual(results["Sample Standard Deviation"], 22.0052, places=4)
        
        # Test sample counts
        self.assertEqual(results["Number of Pairs"], 25)
        self.assertEqual(results["Number of Pairs with a Sign"], 21)
        self.assertEqual(results["Number of times Sample  is Larger"], 11)
        self.assertEqual(results["Number of times Sample is Smaller"], 10)
        self.assertEqual(results["Number of ties"], 4)
        
        # Test Wilcoxon method results
        self.assertEqual(results["Sum of the Positive Ranks (without ties)"], 131.0)
        self.assertEqual(results["Sum of the Negative Ranks (without ties)"], 100.0)
        self.assertEqual(results["Wilcoxon Mean W (Without ties)"], 115.5)
        self.assertAlmostEqual(results["Wilcoxon Standard Deviation"], 28.7380, places=4)
        self.assertAlmostEqual(results["Wilcoxon Z"], 0.5394, places=4)
        self.assertAlmostEqual(results["Wilcoxon Z With Normal Approximation (Continuiety Correction)"], 0.5220, places=4)
        self.assertAlmostEqual(results["Wilcoxon p-value"], 0.5896, places=4)
        self.assertAlmostEqual(results["Wilcoxon p-value with Normal Approximation (Continuiety Correction)"], 0.6017, places=4)
        
        # Test correlation coefficients
        self.assertAlmostEqual(results["Matched Pairs Rank Biserial Correlation (Ignoring Ties)"], 0.1342, places=4)
        self.assertAlmostEqual(results["Z-based Rank Biserial Correlation (Wilcoxon Method)"], 0.1177, places=4)
        self.assertAlmostEqual(results["Z-based Corrected Rank Biserial Correlation (Wilcoxon Method)"], 0.1139, places=4)
        
        # Test Pratt method results
        self.assertEqual(results["Sum of the Positive Ranks (with ties)"], 175.0)
        self.assertEqual(results["Sum of the Negative Ranks (with ties)"], 140.0)
        self.assertEqual(results["Pratt MeanW (Considering Ties)"], 157.5)
        self.assertAlmostEqual(results["Pratt Standard Deviation"], 37.0388, places=4)
        self.assertAlmostEqual(results["Pratt Z"], 0.4725, places=4)
        self.assertAlmostEqual(results["Pratt Z with Normal Approximation (Continuiety Correction)"], 0.4590, places=4)
        self.assertAlmostEqual(results["Pratt p-value"], 0.6366, places=4)
        self.assertAlmostEqual(results["Pratt p-value with Normal Approximation (Continuiety Correction)"], 0.6463, places=4)
        
        # Test Pratt correlations
        self.assertAlmostEqual(results["Matched Pairs Rank Biserial Correlation (Considering Ties)"], 0.1077, places=4)
        self.assertAlmostEqual(results["Z-based Rank Biserial Correlation (Pratt Method)"], 0.0945, places=4)
        self.assertAlmostEqual(results["Z-based Corrected Rank Biserial Correlation (Pratt Method)"], 0.0918, places=4)
        
        # Test confidence intervals
        self.assertAlmostEqual(results["Lower CI Matched Pairs Rank Biserial Wilcoxon"], -0.1020, places=4)
        self.assertAlmostEqual(results["Upper CI Matched Pairs Rank Biserial Wilcoxon"], 0.3561, places=4)
        self.assertAlmostEqual(results["Lower CI Z-based Rank Biserial Wilcoxon"], -0.1186, places=4)
        self.assertAlmostEqual(results["Upper CI Z-based Rank Biserial Wilcoxon"], 0.3414, places=4)

if __name__ == '__main__':
    unittest.main() 