import json
import os
import pickle
import shutil
from pycaret.classification import * # Preprocessing, modelling, interpretation, deployment...
import pandas as pd

class Task:
    """A class that stores the configurations to a prediction task."""

    def __init__(self, task_id=None, train_dataset=None,
                 test_dataset=None, target=None, path_to_generator=None,
                 sampling_method=None, pycaret_model=None, run_num=None):
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
            sampling_method (str):
                "uniform" , "original", or "all" (for both uniform and original)
            pycaret_model (str):
                the pycaret classifier ID, this classifier will be trained and tested
            run_num (int):
                the number of runs for the classifier and synthetic data generator.
        """
        self._task_id = task_id
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._target = target
        self._path_to_generator = path_to_generator
        self._sampling_method = sampling_method
        self._pycaret_model = pycaret_model
        self._run_num = run_num

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
    def sampling_method(self):
        return self._sampling_method

    @property
    def pycaret_model(self):
        return self._pycaret_model

    @property
    def run_num(self):
        return self._run_num


def create_tasks(train_dataset="data/train.csv",
                test_dataset="data/test.csv", target="TARGET",
                path_to_generators = "generators/", pycaret_models=None,
                sampling_method="all", run_num=1, output_dir=None):
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
            "uniform" , "original", "baseline", or "all" (for both uniform and original)
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

    sampling_methods = 

    if pycaret_models is None:
        train_data = pd.read_csv(train_dataset)
        test_data = pd.read_csv(test_dataset)
        setup(train_data,
            target = target, 
            test_data = test_data,
            silent=True,
            verbose=False)
        pycaret_models = models().index.to_list()

    generator_paths = []
    generator_name = {}
    for f in os.listdir(path_to_generators):
        file_name, file_type = os.path.splitext(f)
        if file_type == '.pkl':
            generator_path = os.path.join(path_to_generators, f)
            generator_name[generator_path] = file_name
            generator_paths.append(generator_path)

    for classifier in pycaret_models:
        for generator_path in generator_paths:
            for s_m in sampling_methods
            task_id = "{}_{}_{}".format(task_num, generator_name[generator_path], classifier)
            task = Task(task_id=task_id, train_dataset=train_dataset,
                        test_dataset=test_dataset, target=target,
                        path_to_generator=generator_path, sampling_method=sampling_method, 
                        pycaret_model=classifier, run_num=run_num)
            tasks.append(task)
            task_num += 1

    if output_dir is not None:
        if os.path.exists(output_dir):
            #automatically clears output directory
            shutil.rmtree(output_dir) 
        os.mkdir(output_dir)

        for task in tasks:
            task_path = os.path.join(output_dir, task.task_id)
            if os.path.exists(task_path):
                shutil.rmtree(task_path)
            os.mkdir(task_path)
            task.save_as(os.path.join(task_path, 'meta.json'))
    return tasks