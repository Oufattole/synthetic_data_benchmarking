import unittest
import pathlib as pl
import pandas as pd
import os
import numpy as np

import benchmark
import task

class TestBenchmark(unittest.TestCase):
    def results_table(self):
        results_output_path = "results/"
        task_output_path = "tasks/"
        path_to_generators = "generators/"
        tasks = task.create_tasks(train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition",
                    path_to_generators = path_to_generators, pycaret_models=["lr"],
                    task_sampling_method="all", run_num=1, output_dir=task_output_path)
        metrics = ["accuracy"]
        table = benchmark.Results_Table(None, tasks, metrics)
        task_id = "3_default_tvae_original_0_lr_0"
        
        #initialization
        result_df = table.get_df()
        check = result_df.loc[result_df['Task ID'] == task_id]
        self.assertEqual(check.shape[0], 1)
        entry = check["Status"].to_list()[0]
        expected = benchmark.Status.PENDING
        self.assertEqual(entry,expected)
        #update row
        row = [task_id] + ["test"] * (table.result_df.shape[1]-1)
        table.update_row(row)
        result_df = table.get_df()
        check = result_df.loc[result_df['Task ID'] == task_id]
        self.assertEqual(check.shape[0], 1)
        entry = check["Status"].to_list()[0]
        expected = "test"
        self.assertEqual(entry,expected)
        #update row status
        table.update_row_status(task_id, benchmark.Status.SUCCESS)
        result_df = table.get_df()
        check = result_df.loc[result_df['Task ID'] == task_id]
        self.assertEqual(check.shape[0], 1)
        entry = check["Status"].to_list()[0]
        expected = benchmark.Status.SUCCESS
        self.assertEqual(entry,expected)
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

    def test_summary(self):
        np.random.seed(123)
        N = 10
        L1 = list('abcdefghijklmnopqrstu')
        L2 = list('efghijklmnopqrstuvwxyz')
        result_df = pd.DataFrame({'c1':np.random.randint(1000, size=N),
                        'Sampling Method': np.random.randint(5, size=N),
                        'Classifier Name': np.random.randint(5, size=N),
                        'SD Generator Path':np.random.randint(5, size=N)})
        metric = "c1"
        output = benchmark.summarize_top_n(3, metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988,742,595])
        output = benchmark.summarize_sampling_method(3, metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988,742,510])
        output = benchmark.summarize_classifier(3, metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988,742,510])
        output = benchmark.summarize_generator(3, metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988,595,510])
    



if __name__ == '__main__':
    unittest.main()