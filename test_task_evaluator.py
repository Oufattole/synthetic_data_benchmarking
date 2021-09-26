import unittest
import pathlib as pl
import pandas as pd
import os
import shutil

import task
import task_evaluator

class TestTaskEvauator(unittest.TestCase):
    def setUp(self):
        # output directory setup
        output_dir = "tasks"
        task_id = "test_id"
        task_output_dir = os.path.join(output_dir, task_id)
        if os.path.exists(task_output_dir):
            shutil.rmtree(task_output_dir) 
        os.mkdir(task_output_dir) 
        self.output_dir = task_output_dir

    def test_baseline(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="baseline", pycaret_model="lr", run_num=1, output_dir=self.output_dir)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'baseline')
        expected_synth_data_path = os.path.join(test_task.output_dir, "classifier_lr.pkl")
        self.assertTrue(os.path.exists(expected_synth_data_path))
    def test_sampling_method_original_0_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_0", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 20/60')
    def test_sampling_method_original_1_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_1", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 40/60')
    def test_sampling_method_original_2_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_2", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 60/60')
    
    def test_sampling_method_uniform(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="uniform", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'uniform')

if __name__ == '__main__':
    unittest.main()