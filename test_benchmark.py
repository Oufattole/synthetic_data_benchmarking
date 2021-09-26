import unittest
import pathlib as pl
import pandas as pd
import os

import benchmark
import task

class TestBenchmark(unittest.TestCase):
    
    def test_get_scores(self):
        ground_truth = pd.Series({1:"No",2:"No",3:"Yes",4:"Maybe"})
        classifier_predictions = pd.Series({1:"No",2:"No",3:"No",4:"No"})
        classifier_scores = pd.Series({1:.6,2:.6,3:.6,4:.6})
        auc, f1, recall, precision, accuracy, support = benchmark.Task_Evaluator._get_scores(ground_truth, classifier_predictions, classifier_scores)

        self.assertAlmostEqual(1/3, recall[0])
        self.assertEqual({'Maybe': 0.0, 'No': 0.5, 'Yes': 0.0}, precision[1])
        self.assertEqual({'Maybe': 0.0, 'No': 1.0, 'Yes': 0.0}, recall[1])
        self.assertEqual({'Maybe': 1, 'No': 2, 'Yes': 1},support)
        self.assertEqual(auc, None)


        ground_truth = pd.Series({1:"No",2:"No",3:"Yes",4:"Yes"})
        classifier_predictions = pd.Series({1:"No",2:"No",3:"No",4:"No"})
        classifier_scores = pd.Series({1:.6,2:.6,3:.6,4:.6})
        auc, f1, recall, precision, accuracy, support = benchmark.Task_Evaluator._get_scores(ground_truth, classifier_predictions, classifier_scores)

        self.assertAlmostEqual(1/2, recall[0])
        self.assertEqual({'No': 0.5, 'Yes': 0.0}, precision[1])
        self.assertEqual({'No': 1.0, 'Yes': 0.0}, recall[1])
        self.assertEqual({'No': 2, 'Yes': 2},support)
        self.assertEqual(auc, .5)

    def test_benchmark(self):
        pass

if __name__ == '__main__':
    unittest.main()