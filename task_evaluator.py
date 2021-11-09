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
from pycaret import classification
from pycaret import regression

CLASSIFICATION_METRICS = ['auc', 'f1', 'recall', 'precision', 'accuracy']
REGRESSION_METRICS = ["mae", "mse", "r2", "msle"]

class Task_Evaluator():
    def __init__(self, task):
        self.task = task
        self.train_data = pd.read_csv(task.train_dataset)
        self.test_data = pd.read_csv(task.test_dataset)
        generator = sdv.sdv.SDV.load(task.path_to_generator)
        self.sampler = Sampler(task, self.train_data, generator)

    def evaluate_task(self, metrics=None):
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
        all_metrics = REGRESSION_METRICS if self.task.is_regression else CLASSIFICATION_METRICS
        if metrics is None:
            metrics = all_metrics
        
        combined_data, sampling_method_info, score_aggregate = self.sampler.sample_data()

        predictions = self._regression(combined_data) if self.task.is_regression else self._classify(combined_data)
        ground_truth = predictions[self.task.target]
        output_predictions = predictions["Label"]
        output_score = None
        if "Score" in predictions.columns:
            output_score = predictions["Score"]
        scores = self._get_scores(ground_truth, output_predictions, output_score)
        # make dictionary of metric name to score
        
        metric_to_score = {metric:score for metric, score in zip(all_metrics, scores)}
        # record entry
        # results_row = model_column + sample_size_column + score_column + classifier_column + performance_column
        row = [self.task.task_id, self.task.path_to_generator, self.task.pycaret_model,
            sampling_method_info, self.task.run_num]
        for metric in metrics:
            row += [metric_to_score[metric]] # TODO change to append
        return row
    
    def get_sampler_logs(self):
        return self.sampler.logs

    def _get_scores(self, ground_truth, predictions, scores):
        if self.task.is_regression:
            return self._get_regression_scores(ground_truth, predictions, scores)
        else:
            return self._get_classification_scores(ground_truth, predictions, scores)
    
    @classmethod
    def _get_classification_scores(self, ground_truth, classifier_predictions, classifier_score):
        labels = sorted(ground_truth.unique())
        
        precision_avg, recall_avg, f1_avg, _ = sklearn.metrics.precision_recall_fscore_support(ground_truth, classifier_predictions, average="macro", labels = labels)
        precision_label, recall_label, f1_label, support = sklearn.metrics.precision_recall_fscore_support(ground_truth, classifier_predictions, labels = labels)
        accuracy = sklearn.metrics.accuracy_score(ground_truth, classifier_predictions)
        auc = None
        if not classifier_score is None:
            auc = sklearn.metrics.roc_auc_score(ground_truth, classifier_score, average='macro', multi_class="ovr")

        # def convert_labels_lists_to_dict(label_scores):
        #     scores = {label:score for label, score in zip(labels,label_scores)}
        #     return scores

        precision = precision_avg # label specific: convert_labels_lists_to_dict(precision_label)
        recall = recall_avg # label specific: convert_labels_lists_to_dict(recall_label)
        f1 = f1_avg # label convert_labels_lists_to_dict(f1_label)
        # support = convert_labels_lists_to_dict(support)

        return [auc, f1, recall, precision, accuracy, support]
    

    @classmethod
    def _get_regression_scores(self, ground_truth, predictions, classifier_score):
        mae = sklearn.metrics.mean_absolute_error(ground_truth, predictions)
        mse = sklearn.metrics.mean_squared_error(ground_truth, predictions)
        r2 = sklearn.metrics.r2_score(ground_truth, predictions)
        msle=None
        try:
            msle = sklearn.metrics.mean_squared_log_error(ground_truth, predictions)
        except ValueError as e:
            pass
        return [mae, mse, r2, msle]
    
    def _store_classifier(self, classifier_model):
        task_output_dir = self.task.output_dir
        classifier_file_name = f"classifier_{self.task.pycaret_model}"
        classifier_output_path = os.path.join(task_output_dir, classifier_file_name)
        classification.save_model(classifier_model, classifier_output_path)
    def _store_regresser(self, regression_model):
        task_output_dir = self.task.output_dir
        regressor_file_name = f"regressor_{self.task.pycaret_model}"
        regressor_output_path = os.path.join(task_output_dir, regressor_file_name)
        regression.save_model(regression_model, regressor_output_path)

    def _classify(self, combined_data):
        #TODO, check for ordinal and categorical features
        self._classifier_setup(combined_data)
        # Train classifier
        classifier = classification.create_model(self.task.pycaret_model, verbose=False)
        # Store Classifier
        if self.task.output_dir:
            self._store_classifier(classifier)
        # Predict on Test set
        predictions = classification.predict_model(classifier, verbose=False) # TODO get raw_scores for AUC
        return predictions

    def _classifier_setup(self, combined_data):
        classification.setup(combined_data.sample(frac=1), #shuffles the data
            target = self.task.target, 
            test_data = self.test_data,
            fold_strategy = "kfold", # TODO allow more strategies as hyperparam
            silent = True,
            verbose = False)
    def _regression(self, combined_data):
        self._regression_setup(combined_data)
        # Train
        regresser = regression.create_model(self.task.pycaret_model, verbose=False)
        # Store
        if self.task.output_dir:
            self._store_regresser(regresser)
        # Predict on Test set
        predictions = regression.predict_model(regresser, verbose=False) # TODO get raw_scores for AUC
        return predictions
    def _regression_setup(self, combined_data):
        regression.setup(combined_data.sample(frac=1), #shuffles the data
            target = self.task.target, 
            test_data = self.test_data,
            fold_strategy = "kfold", # TODO allow more strategies as hyperparam
            silent = True,
            verbose = False)
