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
        self._store_data(synthetic_data)
        combined_data = pd.concat([self.train_data, synthetic_data])
        return combined_data, sampling_method_info, score_aggregate

    def _store_data(self, synthetic_data):
        pass #TODO

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
        sampling_method = "uniform"
        target = self.task.target
        target_category_frequencies = self.train_data[target].value_counts().to_dict()
        largest_target = self.train_data[target].value_counts().nlargest(n=1).values[0] # get largest target category
        class_to_sample_size = {}
        for class_name in target_category_frequencies:
            sample_size = largest_target - target_category_frequencies[class_name]
            class_to_sample_size[class_name] = sample_size
        all_sampled_data = []

        for class_name, sample_size in class_to_sample_size.items():
            if sample_size > 0:
                conditions = {
                target : class_name
                } 
                try:
                    data = self.generator.sample(sample_size, conditions=conditions)
                    all_sampled_data.append(data)
                except ValueError as e: # handle case where no valid rows could be generated
                    error_to_handle_msg = "No valid rows could be generated with the given conditions."
                    if error_to_handle_msg in str(e):
                        print("################################################################")
                        print(e)#TODO log this case
                        print(f"column_value that couldn't be generated: {class_name}")
                    else:
                        raise e
        if len(all_sampled_data):
            synthetic_data = pd.concat(all_sampled_data)
            score_aggregate = evaluate(synthetic_data, self.train_data, aggregate=True)

            
            return synthetic_data, sampling_method, score_aggregate
        else:
            return None, sampling_method, None

    def _sample_baseline(self):
        sampling_method_info = "baseline"
        return sampling_method_info