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

ORIGINAL_STEPS = 3

class Sampler():
    def __init__(self, sample_method_name, train_data, generator):
        self.sample_method_info = sample_method_name.split("_")
        self.sampling_method = self.sample_method_info[0]
        self.generator = generator
        self.split = "hi"
        self.train_data = train_data

    def sample_data(self):
        if self.sampling_method ==  "original":
            return self._sample_original()
        elif self.sampling_method ==  "uniform":
            return self._sample_uniform()
        elif self.sampling_method ==  "baseline":
            return self._sample_baseline()
        else:
            raise ValueError("for task id {} task.sampling_method_id is {} which is invalid".format(task.task_id, task.sampling_method))

    def _sample_all(self):
        syn_original = self._sample_original() #pdconcat this
        syn_uniform = self._sample_uniform()
        combined_data = pd.concat([self.train_data, syn_original, syn_uniform])

    def _sample_original(self):
        assert(len(self.sample_method_info)==2)
        train_data_size = self.train_data.shape[0]
        step_size = train_data_size // ORIGINAL_STEPS
        steps = int(self.sample_method_info[1])+1
        sample_size = step_size*steps
        synthetic_data = self.generator.sample(sample_size)
        score_aggregate = evaluate(synthetic_data, self.train_data, aggregate=False)
        # score_column = make_score_column(score_aggregate)
        combined_data = pd.concat([self.train_data, synthetic_data])
        sampling_method_info = "original " + str(sample_size) +"/"+ str(train_data_size)
        return combined_data, sampling_method_info

    def _sample_uniform(self, train_data):
        pass # TODO
        target_category_frequencies = self.train_data['status'].value_counts().to_dict()
        largest_target = self.train_data[target].value_counts().nlargest(n=1).values[0] # get largest target category
        for each in target_category_frequency:
            #do rejection sampling
            pass
        
        num_target_samples = self.train_data[train_data[target] == True]#added True here, TODO fix this
        synthetic_data = self.generator.sample(sample_size)
        score_aggregate = evaluate(synthetic_data, train, aggregate=False)
        # score_column = make_score_column(score_aggregate)
        combined_data = pd.concat([self.train_data, synthetic_data])
        sampling_method = "Uniform"
        yield combined_data, sampling_method

    def _sample_baseline(self):
        sampling_method_info = "baseline"
        print("wth")
        return self.train_data, sampling_method_info

    @classmethod
    def get_sample_method_ids(cls, task_sample_method, do_baseline=True):
        if task_sample_method ==  "all":
            yield from cls.get_sample_method_ids("baseline", do_baseline)
            yield from cls.get_sample_method_ids("uniform", do_baseline=False)
            yield from cls.get_sample_method_ids("original", do_baseline=False)
        elif task_sample_method ==  "original":
            yield from cls.get_sample_method_ids("baseline", do_baseline)
            for i in range(ORIGINAL_STEPS):
                yield f"original_{str(i)}"
        elif task_sample_method ==  "uniform":
            yield from cls.get_sample_method_ids("baseline", do_baseline)
            yield "uniform"
        elif task_sample_method == "baseline":
            if do_baseline:
                yield "baseline"
        else:
            raise ValueError("for task id {} task.sampling_method is {} which is invalid".format(task.task_id, task.sampling_method))