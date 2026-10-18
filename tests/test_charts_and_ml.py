import unittest

import pandas as pd

from modules.charts import build_correlation_heatmap, build_histogram
from modules.data_preparation import (
    prepare_features_target,
    split_dataset,
    validate_dataset,
)
from modules.data_utils import make_frequency_table


class ChartEdgeCaseTests(unittest.TestCase):
    def test_constant_numeric_column_heatmap(self):
        df = pd.DataFrame({
            "a": [1, 1, 1, 1],
            "b": [2, 2, 2, 2],
        })
        fig, message = build_correlation_heatmap(df, "corr")
        self.assertIsNone(fig)
        self.assertIn("Not enough valid numeric correlations", message)

    def test_all_missing_histogram(self):
        df = pd.DataFrame({"n": [None, None, None]})
        fig, message = build_histogram(df, "n", "hist")
        self.assertIsNone(fig)
        self.assertIn("no numeric values", message)

    def test_nan_in_correlation_is_dropped(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "constant": [5, 5, 5, 5],
        })
        fig, message = build_correlation_heatmap(df, "corr")
        self.assertIsNotNone(fig)
        self.assertIsNone(message)


class MLValidationTests(unittest.TestCase):
    def test_empty_and_zero_columns(self):
        self.assertTrue(validate_dataset(pd.DataFrame()))
        self.assertTrue(validate_dataset(pd.DataFrame(index=[0, 1])))

    def test_prepare_features_rejects_missing_target(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, None, 0]})
        with self.assertRaises(ValueError):
            prepare_features_target(df, "y")

    def test_split_rejects_invalid_task_type(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "Task type"):
            split_dataset(X, y, task_type="Classifcation")

    def test_split_rejects_missing_target(self):
        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([0, None, 1])
        with self.assertRaisesRegex(ValueError, "missing values"):
            split_dataset(X, y)

    def test_split_rejects_non_finite_test_size(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "finite number"):
            split_dataset(X, y, test_size=float("nan"))


class FrequencyConstantTests(unittest.TestCase):
    def test_grouped_constant_does_not_raise(self):
        table = make_frequency_table(pd.Series([3.0, 3.0]), grouped=True, bins=8)
        self.assertEqual(len(table), 1)


if __name__ == "__main__":
    unittest.main()
