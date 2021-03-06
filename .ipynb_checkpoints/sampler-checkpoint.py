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
import random
import math



class Sampler():
    def __init__(self, task_instance, train_data, generator):
        sample_method_name = task_instance.sampling_method_id
        self.task = task_instance
        self.sample_method_info = sample_method_name.split("_")
        self.sampling_method = self.sample_method_info[0]
        self.generator = generator
        self.train_data = train_data

    def sample_data(self):
        synthetic_data, sampling_method_info, score_aggregate = None, None, None
        if self.sampling_method ==  "original":
            synthetic_data, sampling_method_info, score_aggregate = self._sample_original()
        elif self.sampling_method ==  "uniform":
            synthetic_data, sampling_method_info, score_aggregate = self._sample_uniform()
        elif self.sampling_method ==  "baseline":
            sampling_method_info = self._sample_baseline()
        else:
            raise ValueError("for task id {} task.sampling_method_id is {} which is invalid".format(task.task_id, task.sampling_method))
        synthetic_data_created = (not synthetic_data is None) and synthetic_data.size
        if self.task.output_dir and synthetic_data_created:
            self._store_data(synthetic_data)
        combined_data = pd.concat([self.train_data, synthetic_data])
        return combined_data, sampling_method_info, score_aggregate

    def _store_data(self, synthetic_data):
        task_output_dir = self.task.output_dir
        synth_data_csv_output_dir = os.path.join(task_output_dir, "synthetic_data.csv")
        synthetic_data.to_csv(synth_data_csv_output_dir)


    def _sample_original(self):
        assert(len(self.sample_method_info)==2)
        train_data_size = self.train_data.shape[0]
        step_size = train_data_size // task.ORIGINAL_STEPS
        steps = int(self.sample_method_info[1])+1
        sample_size = step_size*steps
        synthetic_data = self.generator.sample(sample_size)
        score_aggregate = evaluate(synthetic_data, self.train_data, aggregate=True)
        # score_column = make_score_column(score_aggregate)
        
        sampling_method_info = "original " + str(sample_size) +"/"+ str(train_data_size)
        return synthetic_data, sampling_method_info, score_aggregate

    def _sample_uniform(self):
        if self.task.is_regression:
            return self._sample_uniform_regression()
        else:
            return self._sample_uniform_classification()

    def _get_class_to_sample_size(self):
        target = self.task.target
        target_category_frequencies = self.train_data[target].value_counts().to_dict()
        largest_target = self.train_data[target].value_counts().nlargest(n=1).values[0] # get largest target category
        class_to_sample_size = {}
        for class_name in target_category_frequencies:
            sample_size = largest_target - target_category_frequencies[class_name]
            class_to_sample_size[class_name] = sample_size
        return class_to_sample_size

    def _sample_uniform_classification(self):
        sampling_method = "uniform"
        target = self.task.target
        class_to_sample_size = self._get_class_to_sample_size()
        all_sampled_data = []
        for class_name, sample_size in class_to_sample_size.items():
            if sample_size > 0:
                conditions = {
                target : class_name
                } 
                data = self.generator.sample(sample_size, conditions=conditions)
                all_sampled_data.append(data)
        if len(all_sampled_data):
            synthetic_data = pd.concat(all_sampled_data)
            score_aggregate = evaluate(synthetic_data, self.train_data, aggregate=True)

            
            return synthetic_data, sampling_method, score_aggregate
        else:
            return None, sampling_method, None
    def _sample_uniform_regression(self):
        """
        convert self.train_data[self.task.target] continuous column into a
        discrete binned distribution from max to min value of the continuous columns

        then run uniform classification sampling method on it to get the class_to_sample_size
        (bin_to_sample_size in this case)

        then use uniform_bin_draw iteratively and sample iteratively
        """
        sampling_method = self.task.sampling_method_id
        bins = self.task.regression_bins
        original_data = self.train_data
        self.train_data = self.train_data.copy()
        self.train_data[self.task.target] = pd.cut(x=self.train_data[self.task.target], bins=bins)
        #synthetic_data, sampling_method, score_aggregate = self._sample_uniform_classification()
        class_to_sample_size = self._get_class_to_sample_size()
        
        self.train_data = original_data
        dtype = self.train_data[self.task.target].dtypes
        def int_uniform_bin_draw(interval):
            left = interval.left+1 if interval.left.is_integer() else interval.left
            return random.randint(math.ceil(left), math.floor(interval.right))
        def float_uniform_bin_draw(interval):
            return random.uniform(interval.left, interval.right)
        def uniform_bin_draw(interval):
            if pd.api.types.is_integer_dtype(dtype):
                return int_uniform_bin_draw(interval)
            else:
                return float_uniform_bin_draw(interval)
                
        rows = []
        for class_name, sample_size in class_to_sample_size.items():
            for i in range(sample_size):
                target_value = uniform_bin_draw(class_name)
                conditions = {
                self.task.target : target_value
                } 
                data = self.generator.sample(1, conditions=conditions) # get 1 sample
                data[self.task.target] = data[self.task.target].astype(dtype)
                rows.append(data)
        
        synthetic_data = pd.concat(rows)
        assert(self.train_data.dtypes.to_list() == synthetic_data.dtypes.to_list())
        score_aggregate = evaluate(synthetic_data, self.train_data, aggregate=True)
        return synthetic_data, sampling_method, score_aggregate

    def _sample_baseline(self):
        sampling_method_info = "baseline"
        return sampling_method_info