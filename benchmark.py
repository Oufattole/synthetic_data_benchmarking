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

ID_COLUMNS = ["Task ID", "SD Generator Path", "Classifier Name", "Sampling Method", "Run"]

# AUC no available for Multiclass classification
# ex [taskId, /generator, catboost, "Original X samples" or "uniform" or "baseline"]

def benchmark(tasks, metrics=task_evaluator.CLASSIFICATION_METRICS, agnostic_metrics=False,
            output_path='results/'):
    """Run benchmark testing on a set of tasks. Return detailed results of each run stored in a
    DataFrame object.
    Args:
        tasks (list):
            a list of task instances storing meta information of each task.
        metrics (list):
            pycaret classification metrics to record
            a subset of ['Accuracy', 'AUC', 'Recall', 'F1', 'Precision', 'Kappa']
        agnostic_metrics (boolean): TODO add agnostic metrics feature
            whether to record dataset agnostic metrics in results
        output_path (str):
            the dir path to store benchmark results and records of each task.
        save_results (boolean):
            whether to store the benchmark results.
        TODO add option to store models and generated synthetic data
        TODO add regression benchmarking
        TODO add aggregation functions
    Returns:
        pd.DataFrame:
            benchmarking results in detail.
    """
    results = []
    i = 0
    for task in tasks:
        i+=1
        evaluator = task_evaluator.Task_Evaluator(task)
        results.append(evaluator.evaluate_task(metrics=metrics))
    columns = ID_COLUMNS + metrics
    result_df = pd.DataFrame.from_records(results, columns=columns)

    if output_path is not None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        result_df.to_csv(os.path.join(output_path, 'results.csv'))

    return result_df

def _summary_sampling_method(metric, result_df, output_dir):
    """
    returns dataframe of top row (sorted by metric) for each sampling_method in result_df.

    stores output in output_dir
    """

def _summary_classifier(metric, result_df, output_dir):
    """
    returns dataframe of top row (sorted by metric) for each classifier in result_df.

    stores output in output_dir
    """

def _summary_generator(metric, result_df, output_dir):
    """
    returns dataframe of top row (sorted by metric) for each generator in result_df.

    stores output in output_dir
    """

def _summary_top_n(metric, result_df, output_dir, n):
    """
    returns dataframe of top n rows in result_df sorted by metric.

    stores output in output_dir
    """