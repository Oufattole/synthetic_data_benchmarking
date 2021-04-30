import unittest
import benchmark
import pathlib as pl
import pandas as pd
import os

class TestBenchmark(unittest.TestCase):
    def test_split_data(self):
        #make minidataset to test on
        hr_data = pd.read_csv("hr_data.csv")
        small_data = hr_data[["Age", "Attrition", "DistanceFromHome"]].sample(n=100)
        small_data.to_csv("data.csv")
        #check that split_data makes train and test csv
        benchmark.split_data(dataset_path ="data.csv", target_name="Attrition")
        train_path = pl.Path("data/train.csv")
        test_path = pl.Path("data/test.csv")
        self.assertTrue(train_path.is_file())
        self.assertTrue(test_path.is_file())

    def test_create_default_generators(self):
        #check that create_default_generators creates generators
        generators = ["gaussain_copula", "tvae"]
        benchmark.create_default_generators(generators=generators)
        for generator in generators:
            name = "default_" + generator + ".pkl"
            generator_path = pl.Path("generators/" + name)
            self.assertTrue(generator_path.is_file())
            
    def 
if __name__ == '__main__':
    unittest.main()