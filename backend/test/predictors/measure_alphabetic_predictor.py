from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import validation_results
from matchpredictor.predictors.alphabetic_predictor import AlphabeticPredictor
from test.predictors import csv_location


class TestHomePredictor(TestCase):
    def test_accuracy(self) -> None:
        validation_data = validation_results(csv_location, 2019)
        accuracy, _ = Evaluator(AlphabeticPredictor()).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .33)
