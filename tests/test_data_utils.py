import unittest

import pandas as pd

from modules.data_utils import (
    get_column_series,
    get_dataset_summary,
    make_frequency_table,
    make_upload_id,
    sha256_digest,
)


class DatasetSummaryTests(unittest.TestCase):
    def test_empty_dataframe(self):
        summary = get_dataset_summary(pd.DataFrame())
        self.assertEqual(summary["row_count"], 0)
        self.assertEqual(summary["column_count"], 0)
        self.assertEqual(summary["missing_count"], 0)
        self.assertEqual(summary["duplicate_count"], 0)
        self.assertEqual(summary["completeness"], 100.0)
        self.assertEqual(summary["duplicate_percent"], 0.0)

    def test_zero_columns(self):
        df = pd.DataFrame(index=range(3))
        summary = get_dataset_summary(df)
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["column_count"], 0)
        self.assertEqual(summary["completeness"], 100.0)

    def test_duplicate_row_definition(self):
        df = pd.DataFrame({"a": [1, 1, 1, 2]})
        summary = get_dataset_summary(df)
        self.assertEqual(summary["duplicate_count"], int(df.duplicated().sum()))
        self.assertEqual(summary["duplicate_count"], 2)

    def test_missing_and_completeness(self):
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        summary = get_dataset_summary(df)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["total_cells"], 4)
        self.assertAlmostEqual(summary["completeness"], 75.0)


class UploadIdTests(unittest.TestCase):
    def test_sha256_is_stable(self):
        payload = b"abc123"
        self.assertEqual(sha256_digest(payload), sha256_digest(payload))
        self.assertEqual(len(sha256_digest(payload)), 64)

    def test_same_bytes_same_id(self):
        first = make_upload_id("data.csv", b"col\n1")
        second = make_upload_id("data.csv", b"col\n1")
        self.assertEqual(first, second)

    def test_different_content_different_id(self):
        first = make_upload_id("data.csv", b"col\n1")
        second = make_upload_id("data.csv", b"col\n2")
        self.assertNotEqual(first, second)


class ColumnAndFrequencyTests(unittest.TestCase):
    def test_duplicate_column_names(self):
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        series = get_column_series(df, "a")
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(series.iloc[0], 1)

    def test_frequency_table_all_missing(self):
        table = make_frequency_table(pd.Series([None, None]))
        self.assertTrue(table.empty)

    def test_frequency_table_constant_numeric(self):
        table = make_frequency_table(pd.Series([5, 5, 5]), grouped=True, bins=10)
        self.assertFalse(table.empty)
        self.assertEqual(int(table["Frequency"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
