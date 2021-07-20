import unittest
import pathlib as pl
import pandas as pd
import os

import task
import task_evaluator

class TestTaskEvauator(unittest.TestCase):
    def test_baseline(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="baseline", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        results = evaluator.evaluate_task()
        sampling_methods = [each[3] for each in results]
        # print(sampling_methods)
        self.assertEqual(sampling_methods, ['baseline'])
    def test_sampling_method_original_0_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_0", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        results = evaluator.evaluate_task()
        sampling_methods = [each[3] for each in results]
        self.assertEqual(sampling_methods, ['original 20/60'])
    def test_sampling_method_original_1_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_1", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        results = evaluator.evaluate_task()
        sampling_methods = [each[3] for each in results]
        self.assertEqual(sampling_methods, ['original 40/60'])
    def test_sampling_method_original_2_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_2", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        results = evaluator.evaluate_task()
        sampling_methods = [each[3] for each in results]
        self.assertEqual(sampling_methods, ['original 60/60'])

if __name__ == '__main__':
    unittest.main()