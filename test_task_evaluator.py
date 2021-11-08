import unittest
import pathlib as pl
import pandas as pd
import os
import shutil

import task
import task_evaluator

class TestTaskEvauator(unittest.TestCase):
    def setUp(self):
        # output directory setup
        output_dir = "tasks"
        task_id = "test_id"
        task_output_dir = os.path.join(output_dir, task_id)
        if os.path.exists(task_output_dir):
            shutil.rmtree(task_output_dir) 
        os.mkdir(task_output_dir) 
        self.output_dir = task_output_dir

    def test_baseline(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="baseline", pycaret_model="lr", run_num=1, output_dir=self.output_dir)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'baseline')
        expected_synth_data_path = os.path.join(test_task.output_dir, "classifier_lr.pkl")
        self.assertTrue(os.path.exists(expected_synth_data_path))
    def test_sampling_method_original_0_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_0", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 20/60')
    def test_sampling_method_original_1_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_1", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 40/60')
    def test_sampling_method_original_2_2(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="original_2", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 60/60')
    
    def test_sampling_method_uniform(self):
        output_dir = "tasks"
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="uniform", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'uniform')
    
    def test_get_scores(self):
        # ground_truth = pd.Series({1:"No",2:"No",3:"Yes",4:"Maybe"})
        # classifier_predictions = pd.Series({1:"No",2:"No",3:"No",4:"No"})
        # classifier_scores = pd.Series({1:.6,2:.6,3:.6,4:.6})
        # auc, f1, recall, precision, accuracy, support = task_evaluator.Task_Evaluator._get_scores(ground_truth, classifier_predictions, classifier_scores)

        # self.assertAlmostEqual(1/3, recall[0])
        # self.assertEqual({'Maybe': 0.0, 'No': 0.5, 'Yes': 0.0}, precision[1])
        # self.assertEqual({'Maybe': 0.0, 'No': 1.0, 'Yes': 0.0}, recall[1])
        # self.assertEqual({'Maybe': 1, 'No': 2, 'Yes': 1},support)
        # self.assertEqual(auc, None)


        # ground_truth = pd.Series({1:"No",2:"No",3:"Yes",4:"Yes"})
        # classifier_predictions = pd.Series({1:"No",2:"No",3:"No",4:"No"})
        # classifier_scores = pd.Series({1:.6,2:.6,3:.6,4:.6})
        # auc, f1, recall, precision, accuracy, support = task_evaluator.Task_Evaluator._get_scores(ground_truth, classifier_predictions, classifier_scores)

        # self.assertAlmostEqual(1/2, recall[0])
        # self.assertEqual({'No': 0.5, 'Yes': 0.0}, precision[1])
        # self.assertEqual({'No': 1.0, 'Yes': 0.0}, recall[1])
        # self.assertEqual({'No': 2, 'Yes': 2}, support)
        # self.assertEqual(auc, .5)
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id="36_none_baseline_svm_0", train_dataset="data/train.csv",
                            test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                            sampling_method_id="baseline", pycaret_model="lr", run_num=0)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        df = evaluator.train_data
        for i in range(10):
            df2 = {column:i for column in df.columns}
            df2[df.columns[-1]] = "Maybe"
            df = df.append(df2, ignore_index = True)
        # print(original_data.columns)
        # df = pd.concat([pd.DataFrame([i, i, i, i, "Maybe"],
        #     columns=original_data.columns, ignore_index=True) for i in range(10)])
        evaluator.train_data = df
        # TODO fix AUC here, then check benchmarking outputs results, create aggregation functions, then benchmark
        # evaluator.train_data = pd.concat()
        result = evaluator.evaluate_task()
        self.assertTrue(not result[0] is None)

    def test_svm_classifier(self):
        #TestTaskEvauator.test_svm_classifier
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id="36_none_baseline_svm_0", train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition", path_to_generator=path_to_generator,
                 sampling_method_id="baseline", pycaret_model="svm", run_num=0)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
    
    def test_uniform_regression(self):
        #TestTaskEvauator.test_svm_classifier
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id="36_none_baseline_lr_0", train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Age", path_to_generator=path_to_generator,
                 sampling_method_id="uniform", pycaret_model="lr", run_num=0, is_regression=True)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()      

    def test_uniform_regression_all_state_dataset(self):
        results_output_path = "results/"
        task_output_path = "tasks/"
        path_to_generators = "regression_generators/"
        tasks = task.create_tasks(train_dataset="regression_data/train.csv",
                            test_dataset="regression_data/test.csv", target="charges",
                            path_to_generators = path_to_generators, pycaret_models=["lr", "ridge", "kr"],
                            task_sampling_method="uniform", run_num=1, output_dir=task_output_path, is_regression=True)
        test_task = tasks[0]
        # run benchmark on tasks
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()  

if __name__ == '__main__':
    unittest.main()