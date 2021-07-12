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


SDV_GENGERATORS = ["ct_gan", "gaussain_copula", "copula_gan", "tvae"]
CLASSIFICATION_METRICS = ['auc', 'f1', 'recall', 'precision', 'accuracy', 'support']
ORIGINAL_SAMPLE_METHOD_Granularity = 2
ID_COLUMNS = ["Task ID", "SD Generator Path", "Classifier Name", "Sampling Method", "Run"]
# AUC no available for Multiclass classification
#ex [taskId, /generator, catboost, "Original X samples" or "uniform" or "baseline"]

def split_data(dataset_path ="data.csv", output_directory="data", train_filename = "train.csv", test_filename = "test.csv", target_name="TARGET"):
    """Split real data into train and test set
    Args:
        dataset_path:
            path to dataset csv
        output_directory:
            relative or absolute directory to store outputs in
        train_filename:
            filename to use for train data split, include .csv
        test_filename:
            filename to use for test data split, include .csv
        target_name:    name of target column in dataset
    Return:
        list:
            [train_dataset output path, test_dataset output path] 
    """
    data = pd.read_csv(dataset_path)
    data.head()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train, test, target_train, target_test = train_test_split(data.drop(target_name, axis = 1), data[target_name], test_size = 0.4, random_state = 42)
    train[target_name] = target_train
    test[target_name] = target_test
    train.to_csv(os.path.join(output_directory, train_filename))
    test.to_csv(os.path.join(output_directory, test_filename))


def create_default_generators(train_dataset="data/train.csv", generators=SDV_GENGERATORS, output_directory="generators"):
    """Create SDV Generators with default hyperparameters
    Args:
        train_dataset:
            the path to train dataset
        generators:
            the sdv generators to use, must be a subset of SDV_GENGERATORS
        output_directory:
            the directory to store generator pickle files
    Returns:
        list:
            a list of output_paths of the generator pickle files
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    data = pd.read_csv(train_dataset)
    models = []
    name_to_model = \
    {"ct_gan":CTGAN, "gaussain_copula":GaussianCopula, "copula_gan":CopulaGAN, "tvae":TVAE}
    for name in generators:
        assert(name in SDV_GENGERATORS)
        models.append(name_to_model[name])
    trained_models = []
    output_paths = []
    for model, name in zip(models, generators):
        model_instance = model()
        model_instance.fit(data)
        output_path = os.path.join(output_directory, "default_" + name + ".pkl")
        model_instance.save(output_path)
        output_paths.append(output_path)
    return output_paths

def benchmark(tasks, metrics=CLASSIFICATION_METRICS, agnostic_metrics=False,
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

class Task_Evaluator():
    def __init__(self, task):
        self.task = task
        self.train_data = pd.read_csv(task.train_dataset)
        self.test_data = pd.read_csv(task.test_dataset)
        self.generator = sdv.sdv.SDV.load(task.path_to_generator)
        
    def _sample_data(self):
        #get starting data
        yield self.train_data, "Baseline" #TODO only do one baseline
        #get data combined withs ampling methods
        if self.task.sampling_method ==  "all":
            yield from self._sample_all()
        elif self.task.sampling_method ==  "original":
            yield from self._sample_original()
        elif self.task.sampling_method ==  "uniform":
            yield from self._sample_uniform()
        else:
            raise ValueError("for task id {} task.sampling_method is {} which is invalid".format(task.task_id, task.sampling_method))

    def _sample_all(self):
        yield from self._sample_original()
        yield from self._sample_uniform()

    def _sample_original(self):
        train_data_size = self.train_data.shape[0]
        max_generated_samples = train_data_size + 1
        step_size = train_data_size // ORIGINAL_SAMPLE_METHOD_Granularity
        for sample_size in range(step_size, max_generated_samples, step_size):
            synthetic_data = self.generator.sample(sample_size)
            score_aggregate = evaluate(synthetic_data, self.train_data, aggregate=False)
            # score_column = make_score_column(score_aggregate)
            combined_data = pd.concat([self.train_data, synthetic_data])
            sampling_method = "Original " + str(sample_size) +"/"+ str(train_data_size)
            yield combined_data, sampling_method

    def _sample_uniform(self):
        pass # TODO
        target_category_frequencies = self.train_data['status'].value_counts().to_dict()
        largest_target = self.train_data[target].value_counts().nlargest(n=1).values[0] # get largest target category
        for each in target_category_frequency:
            #do rejection sampling
            pass
        
        num_target_samples = train_data[train_data[target] == True]#added True here, TODO fix this
        synthetic_data = self.generator.sample(sample_size)
        score_aggregate = evaluate(synthetic_data, train, aggregate=False)
        # score_column = make_score_column(score_aggregate)
        combined_data = pd.concat([self.train_data, synthetic_data])
        sampling_method = "Uniform"
        yield combined_data, sampling_method

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
        results = []
        model = self.generator
        for run in range(self.task.run_num):
            for combined_data, sampling_method in self._sample_data():
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
                    sampling_method, run]
                for metric in metrics:
                    row += [metric_to_score[metric]] # TODO change to append
                results.append(row)
        return results
        
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

    def _classify(self, combined_data):
        #TODO, check for ordinal and categorical features
        self._classifier_setup(combined_data)
        # Train classifier
        classifier = create_model(self.task.pycaret_model, verbose=False)
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

