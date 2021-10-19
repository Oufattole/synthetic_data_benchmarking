import json
import os
import pickle
import shutil
import pandas as pd
from pycaret import classification
from pycaret import regression

ORIGINAL_STEPS = 3
SAMPLING_METHODS = ["all", "original", "uniform", "baseline"]

def get_sample_method_ids(sample_method):
    if sample_method != "baseline":
        yield "baseline"
    yield from _get_sample_method_ids_no_baseline(sample_method)

def _get_sample_method_ids_no_baseline(sample_method):
    if sample_method ==  "all":
        yield from _get_sample_method_ids_no_baseline("uniform")
        yield from _get_sample_method_ids_no_baseline("original")
    elif sample_method ==  "original":
        for i in range(ORIGINAL_STEPS):
            yield f"original_{str(i)}"
    elif sample_method ==  "uniform":
        yield "uniform"
    elif sample_method == "baseline":
        yield "baseline"
    else:
        raise ValueError("for task id {} task.sampling_method is {} which is invalid".format(task.task_id, task.sampling_method))

class Task:
    """A class that stores the configurations to a prediction task."""
    def __init__(self, task_id=None, train_dataset=None,
                 test_dataset=None, target=None, path_to_generator=None,
                 sampling_method_id=None, pycaret_model=None, run_num=None,
                 output_dir=None, is_regression=False, regression_bins=5):
        """Create a task configuration object from a list of settings.
        Args:
            task_id (str):
                an identifier for the task.
            train_dataset (str):
                the path where train dataset csv is stored
            test_dataset (str):
                the path where test dataset csv is stored
            target (str):
                the name of the target column in train_dataset and test_dataset
            path_to_generator (str)
                the path where the generator is stored
            sampling_method_id (str):
                unique sampling method id: "uniform" , "original", or "baseline"
            pycaret_model (str):
                the pycaret classifier ID, this classifier will be trained and tested
            run_num (int):
                the number of runs for the classifier and synthetic data generator.
            is_regression (bool):
                perform regression to predict target (default is classification)
        """
        self._task_id = task_id
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._target = target
        self._path_to_generator = path_to_generator
        self._sampling_method_id = sampling_method_id
        self._pycaret_model = pycaret_model
        self._run_num = run_num
        self._output_dir = output_dir
        self._is_regression = is_regression
        self._regression_bins = regression_bins

    def __str__(self):
        description_str = ""
        ordered_items = list(sorted(self.__dict__.items(), key=lambda x: x[0]))
        for k, v in ordered_items:
            description_str += "{:<20} {}\n".format(k, v)
        return description_str

    def __repr__(self):
        description_str = ""
        ordered_items = list(sorted(self.__dict__.items(), key=lambda x: x[0]))
        for k, v in ordered_items:
            description_str += "{:<20} {}\n".format(k, v)
        return description_str

    def save_as(self, file_path):
        """Save the task configurations to the given address.
        Args:
            file_path (str):
                the path to store the configurations.
        """
        _, file_type = os.path.splitext(file_path)
        if file_type == '.pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
        elif file_type == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.__dict__, f)
        else:
            raise ValueError("file_type should be either \"pkl\" or \"json\"")

    @staticmethod
    def load(file_path):
        with open(file_path, 'f') as f:
            attr_dict = json.load(f)
        return Task(**attr_dict)
    
    @property
    def task_id(self):
        return self._task_id

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, train_dataset):
        self._train_dataset = train_dataset
    
    @property
    def test_dataset(self):
        return self._test_dataset
    
    @test_dataset.setter
    def test_dataset(self, test_dataset):
        self._test_dataset = test_dataset

    @property
    def target(self):
        return self._target

    @property
    def path_to_generator(self):
        return self._path_to_generator
    
    @path_to_generator.setter
    def path_to_generator(self, test_dataset):
        self._path_to_generator = path_to_generator
    
    @property
    def sampling_method_id(self):
        return self._sampling_method_id

    @property
    def pycaret_model(self):
        return self._pycaret_model

    @property
    def run_num(self):
        return self._run_num
    
    @property
    def output_dir(self):
        return self._output_dir

    @property
    def is_regression(self):
        return self._is_regression

    @property
    def regression_bins(self):
        return self._regression_bins


def create_tasks(train_dataset="data/train.csv",
                test_dataset="data/test.csv", target="TARGET",
                path_to_generators = "generators/", pycaret_models=None,
                task_sampling_method="all", run_num=1, output_dir=None,
                is_regression=False, regression_bins=5):
    """Create a list of benchmark task objects.
    
    Args:
        train_dataset (str):
            the directory of training dataset csv file
        test_dataset (str):
            the directory of test dataset csv file
        target (str)
            the name of the target column in the train and test dataset
            (must be the same for both datasets)
        path_to_generators (str)
            the directory of generators
        pycaret_models (list):
           list of strings of pycaret classification models to use, if None runs all.
        sampling_method (str):
            "uniform" , "original", "baseline" (no sampling), or "all" (for both uniform and original)
        run_num (int):
            the number of times to generate a sample and test a classifier on it.
        output_dir (str):
            the path to store the task configurations.

    Returns:
        list:
            a list of Task objects that store the benchmarking task        
            configurations.
    """
    task_num = 0
    tasks = []
    
    if pycaret_models is None:
        train_data = pd.read_csv(train_dataset)
        test_data = pd.read_csv(test_dataset)
        pycaret_functions = classification
        if is_regression:
            pycaret_functions = regression
        pycaret_functions.setup(train_data,
            target = target, 
            test_data = test_data,
            silent=True,
            verbose=False)
        pycaret_models = pycaret_functions.models().index.to_list()

    generator_paths = []
    generator_name = {}
    for f in os.listdir(path_to_generators):
        file_name, file_type = os.path.splitext(f)
        if file_type == '.pkl':
            generator_path = os.path.join(path_to_generators, f)
            generator_name[generator_path] = file_name
            generator_paths.append(generator_path)

    if output_dir is not None:
        if os.path.exists(output_dir):
            #automatically clears output directory
            shutil.rmtree(output_dir) 
        os.mkdir(output_dir)

    def create_task(gen_name, task_num, classifier, generator_path,
        sampling_method_id, run_num, output_dir):
        task_id = "{}_{}_{}_{}_{}".format(task_num, gen_name, 
                                        sampling_method_id, classifier,
                                        run_num)
        task_output_dir = None
        if output_dir is not None:
            task_output_dir = os.path.join(output_dir, task_id)
            os.mkdir(task_output_dir)

        task_instance = Task(task_id=task_id, train_dataset=train_dataset,
                    test_dataset=test_dataset, target=target,
                    path_to_generator=generator_path, sampling_method_id=sampling_method_id, 
                    pycaret_model=classifier, run_num=run_num, output_dir=task_output_dir,
                    is_regression=is_regression, regression_bins=regression_bins)
        
        if output_dir is not None:
            task_instance.save_as(os.path.join(task_output_dir, 'meta.json'))
        return task_instance
        

    for classifier in pycaret_models:
        for sampling_method_id in get_sample_method_ids(task_sampling_method):
            for run in range(run_num):
                if sampling_method_id == "baseline":
                    gen_name = "none"
                    task_instance = create_task(gen_name, task_num, classifier,
                        generator_path, sampling_method_id, run, output_dir)
                    tasks.append(task_instance)
                    task_num += 1
                else:
                    for generator_path in generator_paths:
                        gen_name = generator_name[generator_path]
                        task_instance = create_task(gen_name, task_num, classifier,
                            generator_path, sampling_method_id, run, output_dir)
                        tasks.append(task_instance)
                        task_num += 1
                

    
    return tasks

