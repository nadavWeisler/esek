import unittest
from ESEK_Functions.Calculator.1_OneSampleMean.1A_OneSampleZ import OneSampleZ

class TestOneSampleZ(unittest.TestCase):

    def test_one_sample_from_z_score(self):
        params = {
            "Z-score": 1.96,
            "Sample Size": 30,
            "Confidence Level": 95
        }
        results = OneSampleZ.one_sample_from_z_score(params)
        self.assertAlmostEqual(results["Cohen's d"], 0.3578, places=4)
        self.assertAlmostEqual(results["p-value"], 0.0500, places=4)

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main() 