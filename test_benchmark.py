import unittest
import pathlib as pl
import pandas as pd
import os

import benchmark
import task

class TestBenchmark(unittest.TestCase):

    def test_benchmark(self):
        # create tasks
        results_output_path = "results/"
        task_output_path = "tasks/"
        path_to_generators = "generators/"
        tasks = task.create_tasks(train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition",
                    path_to_generators = path_to_generators, pycaret_models=["lr"],
                    task_sampling_method="all", run_num=1, output_dir=task_output_path)
        # run benchmark on tasks
        results = benchmark.benchmark(tasks, agnostic_metrics=False, output_path=results_output_path)
        #check results
        self.assertEqual(len(tasks), results.shape[0])
        self.assertTrue(os.path.exists(os.path.join(results_output_path, 'results.csv')))

        
if __name__ == '__main__':
    unittest.main()