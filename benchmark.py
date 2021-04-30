from pycaret.classification import * # Preprocessing, modelling, interpretation, deployment...
import pandas as pd # Basic data manipulation
from sklearn.model_selection import train_test_split # Data split
from sdv.tabular import CopulaGAN, GaussianCopula, CTGAN, TVAE # Synthetic data
from sdv.evaluation import evaluate # Evaluate synthetic data
import sdmetrics
import sklearn
import os

SDV_GENGERATORS = ["ct_gan", "gaussain_copula", "copula_gan", "tvae"]

def split_data(dataset_path ="data.csv", output_directory="data", train_filename = "train.csv", test_filename = "test.csv", target_name="TARGET"):
    """
        Split real data into training + test set
    inputs---
    dataset_path:   path to dataset csv
    output_directory:      relative or absolute directory to store outputs in
    train_filename: filename to use for train data split, include .csv
    test_filename:  filename to use for test data split, include .csv
    target_name:    name of target column in dataset
    return---
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
    """
       Create SDV Generators with default hyperparameters
    inputs---
    train_dataset:     path to train dataset
    generators:        which sdv generators to use, must be a subset of SDV_GENGERATORS
    output_directory:  directory to store generator pickle files
    return---
    list of output_paths of the generator pickle files
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

class Task():
    def __init__(self, train_dataset, test_dataset, target_column, pycaret_model, sampling_method, run_num):
        pass
    def store_task_as_json():
        pass
    def load_task_from_json():
        pass
    
def create_benchmark_tasks(train_dataset="data/train.csv", test_dataset="data/test.csv", target_column="TARGET", \
                           pycaret_models=None, sampling_methods=[None, "uniform", "original"], run_num=1):
    """Create a list of benchmark task objects.
    
    Args:
        train_dataset (str):
            Directory of training dataset csv file
        test_dataset (str):
            Directory of test dataset csv file
        target_name (str)
            name of target column in train and test dataset (must be same
            for both datasets.
        pycaret_models (list):
           list of strings of pycaret models to use, if None runs all.
        sampling_method (list):
            list of sampling methods
        run_num (int):
            the number of times to generate a sample and test a classifier     
            on it.

    Returns:
        list:
            a list of Task objects that store the benchmarking task        
            configurations.
    """
    train = pd.read_csv(train_dataset)
    test = pd.read_csv(test_dataset)
    if pycaret_models==None:
        pycaret_models = models()["ID"].tolist()
    print(pycaret_models)