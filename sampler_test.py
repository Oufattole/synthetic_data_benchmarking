import unittest
import pathlib as pl
import pandas as pd
import os
import sdv.sdv


from sampler import Sampler

class TestSampler(unittest.TestCase):
    
    def test_sampling_tasks(self):
        sampling_methods = ["all", "original", "uniform", "baseline"]
        all_expected = ["baseline", "uniform", "original_0", "original_1", "original_2"]
        original_expected = ["baseline", "original_0", "original_1", "original_2"]
        uniform_expected = ["baseline", "uniform"]
        baseline_expected = ["baseline"]
        expected_outputs = [all_expected,original_expected, uniform_expected, baseline_expected]
        for task_sample_method, expected in zip(sampling_methods, expected_outputs):
            sampling_output = [each for each in Sampler.get_sample_method_ids(task_sample_method)]
            self.assertEqual(sampling_output, expected)
    
    def test_original_sampling_method(self):
        generator = sdv.sdv.SDV.load("generators/default_gaussain_copula.pkl")
        train_data = pd.read_csv("data/train.csv")

        sampler = Sampler("original_0", train_data, generator)
        comb_data, name = sampler.sample_data()
        self.assertEqual(80, comb_data.shape[0])
        self.assertEqual("original 20/60", name)

        sampler = Sampler("original_1", train_data, generator)
        comb_data, name = sampler.sample_data()
        self.assertEqual(100, comb_data.shape[0])
        self.assertEqual("original 40/60", name)

        sampler = Sampler("original_2", train_data, generator)
        comb_data, name = sampler.sample_data()
        self.assertEqual(120, comb_data.shape[0])
        self.assertEqual("original 60/60", name)

    def test_baseline_sampling_method(self):
        generator = sdv.sdv.SDV.load("generators/default_gaussain_copula.pkl")
        train_data = pd.read_csv("data/train.csv")

        sampler = Sampler("baseline", train_data, generator)
        comb_data, name = sampler.sample_data()
        self.assertEqual(60, comb_data.shape[0])
        self.assertEqual("baseline", name)

if __name__ == '__main__':
    unittest.main()