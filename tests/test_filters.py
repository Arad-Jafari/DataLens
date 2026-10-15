import unittest
from datetime import date

import pandas as pd

from modules.filters import (
    apply_filters,
    combined_filter_mask,
    infer_column_type,
    parse_numeric,
    quick_search_mask,
    validate_filter_drafts,
)


class ParseAndTypeTests(unittest.TestCase):
    def test_parse_numeric(self):
        self.assertEqual(parse_numeric("1,250.5"), 1250.5)
        self.assertIsNone(parse_numeric("not-a-number"))
        self.assertIsNone(parse_numeric(" "))
        self.assertIsNone(parse_numeric(None))

    def test_infer_types(self):
        self.assertEqual(infer_column_type(pd.Series([True, False])), "text")
        self.assertEqual(infer_column_type(pd.Series([1, 2, 3])), "numeric")
        self.assertEqual(
            infer_column_type(pd.to_datetime(["2024-01-01", "2024-01-02"])),
            "date",
        )
        self.assertEqual(infer_column_type(pd.Series([None, None])), "text")
        self.assertEqual(infer_column_type(pd.Series(["10", "20", "30"])), "numeric")


class ValidationTests(unittest.TestCase):
    def test_invalid_numeric_value(self):
        drafts = [{
            "column": "n",
            "kind": "numeric",
            "operator": "Equals",
            "value": None,
        }]
        validated, errors = validate_filter_drafts(drafts, ["n"])
        self.assertEqual(validated, [])
        self.assertTrue(errors)

    def test_reversed_numeric_range(self):
        drafts = [{
            "column": "n",
            "kind": "numeric",
            "operator": "Between",
            "value": [10, 1],
        }]
        validated, errors = validate_filter_drafts(drafts, ["n"])
        self.assertEqual(validated, [])
        self.assertIn("minimum exceeds maximum", errors[0])

    def test_reversed_date_range(self):
        drafts = [{
            "column": "d",
            "kind": "date",
            "operator": "Between",
            "value": [date(2024, 2, 1), date(2024, 1, 1)],
        }]
        validated, errors = validate_filter_drafts(drafts, ["d"])
        self.assertEqual(validated, [])
        self.assertIn("start date exceeds end date", errors[0])

    def test_missing_operator_skips_value(self):
        drafts = [{
            "column": "n",
            "kind": "numeric",
            "operator": "Is missing",
            "value": "ignored",
        }]
        validated, errors = validate_filter_drafts(drafts, ["n"])
        self.assertEqual(errors, [])
        self.assertEqual(validated[0]["value"], None)


class FilterMaskTests(unittest.TestCase):
    def test_duplicate_index_does_not_duplicate_rows(self):
        df = pd.DataFrame(
            {"city": ["Paris", "Lyon", "Paris"], "n": [1, 2, 3]},
            index=["a", "a", "b"],
        )
        filtered = apply_filters(
            df,
            [{
                "column": "city",
                "kind": "text",
                "operator": "Equals",
                "value": "Paris",
            }],
        )
        self.assertEqual(len(filtered), 2)
        self.assertListEqual(filtered["n"].tolist(), [1, 3])

    def test_and_or_logic(self):
        df = pd.DataFrame({"n": [1, 5, 9], "label": ["a", "b", "a"]})
        conditions = [
            {
                "column": "n",
                "kind": "numeric",
                "operator": "Greater than",
                "value": 3,
            },
            {
                "column": "label",
                "kind": "text",
                "operator": "Equals",
                "value": "a",
            },
        ]
        and_result = apply_filters(df, conditions, "AND")
        or_result = apply_filters(df, conditions, "OR")
        self.assertEqual(len(and_result), 1)
        self.assertEqual(len(or_result), 3)

    def test_numeric_invalid_strings_are_not_matched(self):
        df = pd.DataFrame({"n": ["10", "abc", "12"]})
        filtered = apply_filters(
            df,
            [{
                "column": "n",
                "kind": "numeric",
                "operator": "Greater than",
                "value": 10,
            }],
        )
        self.assertListEqual(filtered["n"].tolist(), ["12"])

    def test_all_missing_column(self):
        df = pd.DataFrame({"n": [None, None]})
        missing = apply_filters(
            df,
            [{
                "column": "n",
                "kind": "numeric",
                "operator": "Is missing",
                "value": None,
            }],
        )
        present = apply_filters(
            df,
            [{
                "column": "n",
                "kind": "numeric",
                "operator": "Is not missing",
                "value": None,
            }],
        )
        self.assertEqual(len(missing), 2)
        self.assertEqual(len(present), 0)

    def test_does_not_mutate_source(self):
        df = pd.DataFrame({"n": [1, 2, 3]})
        original = df.copy()
        apply_filters(
            df,
            [{
                "column": "n",
                "kind": "numeric",
                "operator": "Greater than",
                "value": 1,
            }],
        )
        pd.testing.assert_frame_equal(df, original)

    def test_quick_search_is_case_sensitive(self):
        df = pd.DataFrame({"city": ["Paris", "paris"]})
        mask = quick_search_mask(df, "Paris")
        self.assertEqual(int(mask.sum()), 1)

    def test_empty_dataframe_mask(self):
        df = pd.DataFrame({"n": pd.Series(dtype=float)})
        mask = combined_filter_mask(
            df,
            [{
                "column": "n",
                "kind": "numeric",
                "operator": "Is missing",
                "value": None,
            }],
        )
        self.assertEqual(len(mask), 0)


if __name__ == "__main__":
    unittest.main()
