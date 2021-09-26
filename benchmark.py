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
import Task_Evaluator

ID_COLUMNS = ["Task ID", "SD Generator Path", "Classifier Name", "Sampling Method", "Run"]

# AUC no available for Multiclass classification
# ex [taskId, /generator, catboost, "Original X samples" or "uniform" or "baseline"]

def benchmark(tasks, metrics=task_evaluator.CLASSIFICATION_METRICS, agnostic_metrics=False,
            output_path='results/', save_results=True):
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
    if output_path is not None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    performance = []
    for task in tasks:
        if output_path is not None:
            task_output_path = os.path.join(output_path, task.task_id) #TODO consider task level csvs rather than one big csv
        else:
            task_output_path = None
        evaluator = Task_Evaluator(task)
        performance.extend(evaluator.evaluate_task(metrics=metrics))
    columns = ID_COLUMNS + metrics
    result_df = pd.DataFrame.from_records(performance, columns=columns)

    if output_path is not None and save_results:
        result_df.to_csv(os.path.join(output_path, 'details.csv'))

    return result_df

