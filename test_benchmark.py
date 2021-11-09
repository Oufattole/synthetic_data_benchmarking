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

    def test_benchmark_classification(self):
        # create tasks
        results_output_path = "results/"
        task_output_path = "tasks/"
        path_to_generators = "generators/"
        tasks = task.create_tasks(train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition",
                    path_to_generators = path_to_generators, pycaret_models=["lr"],
                    task_sampling_method="all", run_num=1, output_dir=task_output_path)
        # run benchmark on tasks
        result_df, failed_tasks = benchmark.benchmark(tasks, agnostic_metrics=False, output_path=results_output_path)
        #check results
        self.assertEqual(len(tasks), result_df.shape[0])
        self.assertEqual(1, len(failed_tasks))
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
        output = benchmark.summarize_sampling_method(metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988,742,510])
        output = benchmark.summarize_classifier(metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988, 742, 510, 382, 98])
        output = benchmark.summarize_generator(metric, result_df, output_dir=None)
        self.assertEqual(output["c1"].tolist(), [988, 595, 510, 382, 106])
    def test_regression_benchmarking(self):
        results_output_path = "results/"
        task_output_path = "tasks/"
        path_to_generators = "regression_generators/"
        tasks = task.create_tasks(train_dataset="regression_data/train.csv",
                            test_dataset="regression_data/test.csv", target="charges",
                            path_to_generators = path_to_generators, pycaret_models=["lr"],
                            task_sampling_method="all", run_num=1, output_dir=task_output_path, is_regression=True)
        # run benchmark on tasks
        result_df, failed_tasks = benchmark.benchmark(tasks, agnostic_metrics=False,
                                output_path=results_output_path, is_regression=True)
        self.assertEqual(0, len(failed_tasks))

    def test_sampler_logging(self):
        #TestBenchmark.test_sampler_logging
        results_output_path = "results/"
        task_output_path = "tasks/test_id"
        generator_path = "regression_generators/TvaeModel.pkl"
        train_data = "regression_data/train.csv"
        test_data = "regression_data/test.csv"
        target = "charges"
        sampling_method_id="uniform"
        classifier="lr"
        run_num=0
        is_regression=True
        task_output_path = "tasks/"
        tasks = [task.Task(task_id="test_id", train_dataset=train_data,
                    test_dataset=test_data, target=target,
                    path_to_generator=generator_path, sampling_method_id=sampling_method_id, 
                    pycaret_model=classifier, run_num=run_num, output_dir=task_output_path,
                    is_regression=is_regression)]
        # run benchmark on tasks
        result_df, failed_tasks = benchmark.benchmark(tasks, agnostic_metrics=False,
                                output_path=results_output_path, is_regression=True)
        self.assertEqual(0, len(failed_tasks))
    def test_regression_benchmark(self):
        #TestBenchmark.test_regression_benchmark
        results_output_path = "results/"
        task_output_path = "tasks/"
        path_to_generators = "regression_generators/"
        tasks = task.create_tasks(train_dataset="regression_data/train.csv",
                            test_dataset="regression_data/test.csv", target="charges",
                            path_to_generators = path_to_generators, pycaret_models=["lr"],
                            task_sampling_method="uniform", run_num=1, output_dir=task_output_path,
                            is_regression=True)
        # run benchmark on tasks
        result_df, failed_tasks = benchmark.benchmark(tasks, agnostic_metrics=False,
                                    output_path=results_output_path, is_regression=True)
        self.assertEqual(len(failed_tasks), 0)
    



if __name__ == '__main__':
    unittest.main()