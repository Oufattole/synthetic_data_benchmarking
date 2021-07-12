import unittest
import task
import pathlib as pl
import pandas as pd
import os
import json
import shutil

class TestTask(unittest.TestCase):        
    def test_task_init(self):
        task_id = "test_id"
        train_dataset = "data/train.csv"
        test_dataset="data/test.csv"
        target="Attrition"
        path_to_generator="/generators"
        pycaret_model="lr"
        run_num=1
        new_task = task.Task(task_id=task_id, train_dataset=train_dataset,
                            test_dataset=test_dataset, target=target,
                            path_to_generator=path_to_generator,
                            pycaret_model=pycaret_model, run_num=run_num)
        self.assertEqual(task_id, new_task.task_id)
        self.assertEqual(train_dataset, new_task.train_dataset)
        self.assertEqual(test_dataset,  new_task.test_dataset)
        self.assertEqual(target, new_task.target)
        self.assertEqual(path_to_generator, new_task.path_to_generator)
        self.assertEqual(pycaret_model, new_task.pycaret_model)
        self.assertEqual(run_num, new_task.run_num)
        task_path = "tasks/test.json"
        new_task.save_as(task_path)
        self.assertTrue(os.path.exists(task_path))

        task_dict = None
        with open(task_path) as f:
            task_dict = json.load(f)
        self.assertEqual(task_id, task_dict["_task_id"])
        self.assertEqual(train_dataset, task_dict["_train_dataset"])
        self.assertEqual(test_dataset,  task_dict["_test_dataset"])
        self.assertEqual(target, task_dict["_target"])
        self.assertEqual(path_to_generator, task_dict["_path_to_generator"])
        self.assertEqual(pycaret_model, task_dict["_pycaret_model"])
        self.assertEqual(run_num, task_dict["_run_num"])
        os.remove(task_path)

    def test_create_tasks(self):
        output_dir = "tasks"
        path_to_generators = "generators/"
        tasks = task.create_tasks(train_dataset="data/train.csv",
                    test_dataset="data/test.csv", target="Attrition",
                    path_to_generators = path_to_generators, pycaret_models=None,
                    sampling_method="all", run_num=1, output_dir=output_dir)
        
        files = os.listdir(output_dir) # dir is your directory path
        file_count = len(files)
        self.assertEqual(file_count, len(tasks))
        generator_paths = ["generators/default_gaussain_copula.pkl", "generators/default_tvae.pkl"]
        for each in tasks:
            self.assertIn(each.path_to_generator, generator_paths)
        shutil.rmtree(output_dir) 
        os.mkdir(output_dir)       
        
if __name__ == '__main__':
    unittest.main()