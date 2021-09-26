import unittest
import pathlib as pl
import pandas as pd
import os
import sdv.sdv
from task import Task


from sampler import Sampler

class TestSampler(unittest.TestCase):
    def setUp(self):
        generator_path = "generators/default_gaussain_copula.pkl"
        self.generator = sdv.sdv.SDV.load(generator_path)
        train_data_path = "data/train.csv"
        test_data_path = "data/test.csv"
        target = "Attrition"
        self.train_data = pd.read_csv(train_data_path)
        run_num = 1
        classifier = "lr"
        def make_task(sampling_method_id):
            return Task(task_id="test_id", train_dataset=train_data_path,
                    test_dataset=test_data_path, target=target,
                    path_to_generator=generator_path, sampling_method_id=sampling_method_id, 
                    pycaret_model=classifier, run_num=run_num)
        self.make_task = lambda x: make_task(x)
    
    
    
    def test_original_0(self):
        task_original_0 = self.make_task("original_0")
        sampler = Sampler(task_original_0, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(80, combined_data.shape[0])
        self.assertEqual("original 20/60", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
    def test_original_1(self):
        task_original_1 = self.make_task("original_1")
        sampler = Sampler(task_original_1, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(100, combined_data.shape[0])
        self.assertEqual("original 40/60", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
    def test_original_2(self):
        task_original_2 = self.make_task("original_2")
        sampler = Sampler(task_original_2, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(120, combined_data.shape[0])
        self.assertEqual("original 60/60", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)

    def test_baseline_sampling_method(self):
        task_baseline = self.make_task("baseline")
        sampler = Sampler(task_baseline, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(60, combined_data.shape[0])
        self.assertEqual("baseline", sampling_method_info)
        self.assertEqual(score_aggregate, None)
    
    def test_uniform_sampling_method(self):
        task_uniform = self.make_task("uniform")
        sampler = Sampler(task_uniform, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(102, combined_data.shape[0])
        self.assertEqual("uniform", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
        

if __name__ == '__main__':
    unittest.main()