from pycaret.classification import * # Preprocessing, modelling, interpretation, deployment...
import pandas as pd # Basic data manipulation
from sklearn.model_selection import train_test_split # Data split
from sdv.tabular import CopulaGAN, GaussianCopula, CTGAN, TVAE # Synthetic data
from sdv.evaluation import evaluate # Evaluate synthetic data
import sdv.sdv
import sdmetrics
import sklearn
import os
from sampler import Sampler
import task

CLASSIFICATION_METRICS = ['auc', 'f1', 'recall', 'precision', 'accuracy', 'support']

class Task_Evaluator():
    def __init__(self, task):
        self.task = task
        self.train_data = pd.read_csv(task.train_dataset)
        self.test_data = pd.read_csv(task.test_dataset)
        generator = sdv.sdv.SDV.load(task.path_to_generator)
        self.sampler = Sampler(task, self.train_data, generator)

    def evaluate_task(self, metrics=CLASSIFICATION_METRICS):
        """Run benchmark testing on a task. Save intermedia data, trained models, and optimized
        hyperparameters. Return testing results.

        Args:
            task (Task):
                a task instance storing meta information of the task.
            metrics (list)
                a list of strings to identify the metric functions.
            output_path (str):
                a directory path to store the intermedia data, model and hyperparametes.
            agnostic_metrics (boolean):
                whether to record dataset agnostic metrics in results

        Returns:
            list:
                benchmarking results of each run.
        """
        
        combined_data, sampling_method_info, score_aggregate = self.sampler.sample_data()
        predictions = self._classify(combined_data)
        ground_truth = predictions[self.task.target]
        classifier_predictions = predictions["Label"]
        classifier_score = predictions["Score"]
        scores = self._get_scores(ground_truth, classifier_predictions, classifier_score)
        # make dictionary of metric name to score
        metric_to_score = {metric:score for metric, score in zip(CLASSIFICATION_METRICS, scores)}
        # record entry
        # results_row = model_column + sample_size_column + score_column + classifier_column + performance_column
        row = [self.task.task_id, self.task.path_to_generator, self.task.pycaret_model,
            sampling_method_info, self.task.run_num]
        for metric in metrics:
            row += [metric_to_score[metric]] # TODO change to append
        return row
        
    @classmethod
    def _get_scores(self, ground_truth, classifier_predictions, classifier_score):
        labels = sorted(ground_truth.unique())
        binary = len(labels) == 2
        
        precision_avg, recall_avg, f1_avg, _ = sklearn.metrics.precision_recall_fscore_support(ground_truth, classifier_predictions, average="macro", labels = labels)
        precision_label, recall_label, f1_label, support = sklearn.metrics.precision_recall_fscore_support(ground_truth, classifier_predictions, labels = labels)
        accuracy = sklearn.metrics.accuracy_score(ground_truth, classifier_predictions)
        auc = None
        if binary:
            auc = sklearn.metrics.roc_auc_score(ground_truth, classifier_score)

        def convert_labels_lists_to_dict(label_scores):
            scores = {label:score for label, score in zip(labels,label_scores)}
            return scores

        precision = [precision_avg, convert_labels_lists_to_dict(precision_label)]
        recall = [recall_avg, convert_labels_lists_to_dict(recall_label)]
        f1 = [f1_avg, convert_labels_lists_to_dict(f1_label)]
        support = convert_labels_lists_to_dict(support)

        return [auc, f1, recall, precision, accuracy, support]

    def _get_model(self):
        return pick
    def _store_classifier(self, classifier_model):
        task_output_dir = self.task.output_dir
        classifier_file_name = f"classifier_{self.task.pycaret_model}"
        classifier_output_path = os.path.join(task_output_dir, classifier_file_name)
        save_model(classifier_model, classifier_output_path)

    def _classify(self, combined_data):
        #TODO, check for ordinal and categorical features
        self._classifier_setup(combined_data)
        # Train classifier
        classifier = create_model(self.task.pycaret_model, verbose=False)
        # Store Classifier
        if self.task.output_dir:
            self._store_classifier(classifier)
        # Predict on Test set
        predictions = predict_model(classifier, verbose=False) # TODO get raw_scores for AUC
        return predictions

    def _classifier_setup(self, combined_data):
        setup(combined_data.sample(frac=1), #shuffles the data
            target = self.task.target, 
            test_data = self.test_data,
            fold_strategy = "kfold", # TODO allow more strategies as hyperparam
            silent = True,
            verbose = False)