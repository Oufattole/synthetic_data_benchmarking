from pycaret.classification import * # Preprocessing, modelling, interpretation, deployment...
import pandas as pd # Basic data manipulation
from sklearn.model_selection import train_test_split # Data split
from sdv.tabular import CopulaGAN, GaussianCopula, CTGAN, TVAE # Synthetic data
from sdv.evaluation import evaluate # Evaluate synthetic data
import sdv.sdv
import sdmetrics
import sklearn
import os
import pickle
import task
import task_evaluator
import numpy as np

ID_COLUMNS = ["Task ID", "SD Generator Path", "Classifier Name", "Sampling Method", "Run", "Status"]


from enum import Enum
class Status(Enum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    ERRORED = 'ERRORED'

    def get_status(self):
        self.reload()
        return self.status

class Results_Table():
    def __init__(self, output_dir, tasks, metrics):
        columns = ID_COLUMNS + metrics
        self.columns = columns
        results = [[task.task_id, task.path_to_generator, task.pycaret_model,
            task.sampling_method_id, task.run_num, Status.PENDING] + [np.nan]*len(metrics) for task in tasks]
        result_df = pd.DataFrame(results, columns=columns)
        self.result_df = result_df
        self.output_path = os.path.join(output_dir, 'results.csv') if output_dir else None
        self.status = Status.PENDING
        if self.output_path:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            self.result_df.to_csv(self.output_path)
    def update_row(self, row):
        self.result_df.loc[self.result_df["Task ID"]==row[0], self.columns] = row
        if self.output_path:
            self.result_df.to_csv(self.output_path)
    def update_row_status(self, task_id, status):
        self.result_df.loc[self.result_df["Task ID"]==task_id,"Status"] = status
        if self.output_path:
            self.result_df.to_csv(self.output_path)
    def get_df(self):
        return self.result_df

def benchmark(tasks, metrics=task_evaluator.CLASSIFICATION_METRICS, agnostic_metrics=False,
            output_path='results/', summary_metric="accuracy"):
    """Run benchmark testing on a set of tasks. Return detailed results of each run stored in a
    DataFrame object.
    Args:
        tasks (list):
            a list of task instances storing meta information of each task.
        metrics (list):
            pycaret classification metrics to record
            a subset of ['Accuracy', 'AUC', 'Recall', 'F1', 'Precision', 'Kappa']
        agnostic_metrics (boolean):
            whether to record dataset agnostic metrics in results
        output_path (str):
            the dir path to store benchmark results and records of each task.
        save_results (boolean):
            whether to store the benchmark results.
        TODO add regression benchmarking
    Returns:
        pd.DataFrame:
            benchmarking results in detail.
    """
    failed_tasks = []
    results = []
    results_table = None
    results_table = Results_Table(output_path, tasks, metrics)
    for task in tasks:
        results_table.update_row_status(task.task_id, Status.RUNNING)
        evaluator = task_evaluator.Task_Evaluator(task)
        row = None
        try:
            row = evaluator.evaluate_task(metrics=metrics)
        except Exception as error_msg:
            failed_tasks.append(task)
            write_error_log(task.output_dir, error_msg)
            results_table.update_row_status(task.task_id, Status.ERRORED)
        if not row is None:
            center = len(ID_COLUMNS) - 1
            results_table.update_row(row[:center] + [Status.SUCCESS] + row[center:])

        
    # columns = ID_COLUMNS + metrics
    # result_df = pd.DataFrame.from_records(results, columns=columns)
    result_df = results_table.get_df()
    summarize_results(3, "accuracy", result_df, output_path)
    return result_df, failed_tasks

def write_error_log(task_output_dir, error_msg):
    error_log_output_path = os.path.join(task_output_dir, "error_log.txt")
    with open(error_log_output_path, "w") as text_file:
        text_file.write(str(error_msg))

def summarize_sampling_method(metric, result_df, output_dir):
    """
    returns dataframe of top row (sorted by metric) for each sampling_method in result_df.

    stores output in output_dir
    """
    #summary_df = result_df.sort_values(metric, ascending=False).groupby('Sampling Method').head(n)
    summary_df = result_df.groupby('Sampling Method').max(metric).sort_values(metric, ascending=False)
    if output_dir:
        summary_df.to_csv(os.path.join(output_dir, f'summary_sampling_methods_{metric}.csv'))
    return summary_df


def summarize_classifier(metric, result_df, output_dir):
    """
    returns dataframe of top row (sorted by metric) for each classifier in result_df.

    stores output in output_dir
    """
    summary_df = result_df.groupby('Classifier Name').max(metric).sort_values(metric, ascending=False)
    if output_dir:
        summary_df.to_csv(os.path.join(output_dir, f'summary_classifiers_{metric}.csv'))
    return summary_df

def summarize_generator(metric, result_df, output_dir):
    """
    returns dataframe of top row (sorted by metric) for each generator in result_df.

    stores output in output_dir
    """
    summary_df = result_df.groupby('SD Generator Path').max(metric).sort_values(metric, ascending=False)
    if output_dir:
        summary_df.to_csv(os.path.join(output_dir, f'summary_generators_{metric}.csv'))
    return summary_df

def summarize_top_n(n, metric, result_df, output_dir):
    """
    returns dataframe of top n rows in result_df sorted by metric.

    stores output in output_dir
    """
    summary_df = result_df.sort_values(metric, ascending=False).head(n)
    if output_dir:
        summary_df.to_csv(os.path.join(output_dir, f'summary_top_{n}_{metric}.csv'))
    return summary_df

def summarize_results(n, metric, result_df, output_dir):
    summarize_sampling_method(metric, result_df, output_dir)
    summarize_classifier(metric, result_df, output_dir)
    summarize_generator(metric, result_df, output_dir)
    summarize_top_n(n, metric, result_df, output_dir)
