import unittest

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import (
    CLASS_NAMES, DATA, EXAMPLES, FEATURE_NAMES, LOWER_BOUNDS, UPPER_BOUNDS,
    example_values, predict_measurements, train_model, validate_measurements,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = train_model()
        cls.values = example_values(next(iter(EXAMPLES)))

    def test_feature_order_matches_dataset(self):
        self.assertEqual(FEATURE_NAMES, tuple(DATA.feature_names))
        reversed_values = dict(reversed(list(self.values.items())))
        np.testing.assert_array_equal(validate_measurements(reversed_values), DATA.data[[0]])

    def test_original_split_and_real_metrics(self):
        _, x_test, _, y_test = train_test_split(DATA.data, DATA.target, test_size=0.33, random_state=42)
        predicted = self.bundle.estimator.predict(x_test)
        self.assertEqual(self.bundle.accuracy, accuracy_score(y_test, predicted))
        self.assertEqual((self.bundle.train_size, self.bundle.test_size), (381, 188))
        self.assertEqual(int(self.bundle.confusion.sum()), 188)
        self.assertGreater(self.bundle.malignant_recall, 0)

    def test_reproducible_training(self):
        second = train_model()
        self.assertEqual(self.bundle.accuracy, second.accuracy)
        np.testing.assert_array_equal(self.bundle.estimator.theta_, second.estimator.theta_)

    def test_class_mapping(self):
        self.assertEqual(CLASS_NAMES, {0: "Malignant", 1: "Benign"})
        self.assertEqual(tuple(DATA.target_names), ("malignant", "benign"))
        for name in EXAMPLES:
            values = example_values(name)
            result, label = predict_measurements(self.bundle, values)
            self.assertEqual(label, CLASS_NAMES[result])
            self.assertEqual(result, int(self.bundle.estimator.predict(validate_measurements(values))[0]))

    def test_missing_and_extra_features(self):
        for values in [{}, {**self.values, "patient_name": "Do not collect"}, {k: v for k, v in self.values.items() if k != FEATURE_NAMES[0]}]:
            with self.subTest(values=len(values)), self.assertRaises(ValueError):
                validate_measurements(values)

    def test_invalid_values_rejected(self):
        for value in [None, "17.99", "", True, False, float("nan"), float("inf"), -float("inf"), -1, 0, 1e99, object()]:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                validate_measurements({**self.values, FEATURE_NAMES[0]: value})

    def test_every_feature_bounds(self):
        for index, name in enumerate(FEATURE_NAMES):
            for invalid in [LOWER_BOUNDS[index] - 1, UPPER_BOUNDS[index] + 1]:
                with self.subTest(feature=name, value=invalid), self.assertRaises(ValueError):
                    validate_measurements({**self.values, name: invalid})

    def test_public_rows_are_valid(self):
        for row in DATA.data:
            values = dict(zip(FEATURE_NAMES, row))
            np.testing.assert_array_equal(validate_measurements(values)[0], row)

    def test_zero_allowed_for_observed_concavity(self):
        values = {**self.values, "mean concavity": 0.0}
        self.assertEqual(validate_measurements(values)[0, 6], 0)


if __name__ == "__main__":
    unittest.main()
