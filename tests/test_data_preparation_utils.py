import unittest

import numpy as np
import pandas as pd

from modules.data_preparation import (
    DataPreparationHistory,
    change_column_dtype,
    detect_outliers,
    drop_columns_with_missing_values,
    drop_rows_with_missing_values,
    drop_selected_columns,
    fill_missing_values,
    fill_missing_values_with_value,
    get_dummies,
    label_encode,
    manage_duplicates,
    normalize,
    preprocess_split_features,
    rename_column,
    remove_outliers,
    standardize,
)


class OutlierHandlingTests(unittest.TestCase):
    def test_detect_and_remove_outlier_with_duplicate_index(self):
        df = pd.DataFrame(
            {"value": [0, 1, 2, 3, 4, 100], "row": list("abcdef")},
            index=[9, 1, 2, 3, 4, 9],
        )

        outliers = detect_outliers(df, "value")
        cleaned = remove_outliers(df, "value")

        self.assertEqual(outliers["row"].tolist(), ["f"])
        self.assertEqual(cleaned["row"].tolist(), list("abcde"))
        self.assertEqual(df["row"].tolist(), list("abcdef"))

    def test_outlier_detection_ignores_missing_and_infinite_values(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0, np.inf, np.nan]})

        self.assertEqual(detect_outliers(df, "value").index.tolist(), [4])
        self.assertEqual(remove_outliers(df, "value").index.tolist(), [0, 1, 2, 3, 5, 6])

    def test_outlier_detection_rejects_invalid_columns(self):
        with self.assertRaisesRegex(ValueError, "numeric data type"):
            detect_outliers(pd.DataFrame({"value": ["a", "b"]}), "value")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            detect_outliers(pd.DataFrame({"value": [1, 2]}), "missing")


class MissingValueHandlingTests(unittest.TestCase):
    def test_numeric_fill_supports_nullable_integers_and_does_not_mutate(self):
        df = pd.DataFrame({"value": pd.Series([1, None, 2], dtype="Int64")})

        filled = fill_missing_values(df, "value", method="mean")

        self.assertEqual(filled["value"].tolist(), [1.0, 1.5, 2.0])
        self.assertTrue(pd.isna(df.loc[1, "value"]))

    def test_mode_and_directional_fill(self):
        df = pd.DataFrame({"value": ["a", None, "a", None]})

        self.assertEqual(
            fill_missing_values(df, "value", method="mode")["value"].tolist(),
            ["a", "a", "a", "a"],
        )
        with self.assertRaisesRegex(ValueError, "could not fill every missing"):
            fill_missing_values(df, "value", method="bfill")

    def test_all_missing_or_inapplicable_fill_is_reported(self):
        with self.assertRaisesRegex(ValueError, "no values"):
            fill_missing_values(
                pd.DataFrame({"value": [np.nan, np.nan]}), "value"
            )
        with self.assertRaisesRegex(ValueError, "numeric column"):
            fill_missing_values(pd.DataFrame({"value": ["a", None]}), "value")

    def test_fill_with_value_rejects_missing_value(self):
        with self.assertRaisesRegex(ValueError, "must not be missing"):
            fill_missing_values_with_value(
                pd.DataFrame({"value": [None]}), "value", np.nan
            )

    def test_drop_missing_rows_and_columns(self):
        df = pd.DataFrame({"keep": [1, 2], "drop": [None, 3]})

        self.assertEqual(
            list(drop_columns_with_missing_values(df).columns), ["keep"]
        )
        self.assertEqual(len(drop_rows_with_missing_values(df)), 1)
        self.assertEqual(list(df.columns), ["keep", "drop"])


class TransformationTests(unittest.TestCase):
    def test_scalers_return_copy_and_reject_missing_values(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        scaled = normalize(df, "value")
        standardized = standardize(df, "value")

        self.assertEqual(scaled["value"].tolist(), [0.0, 0.5, 1.0])
        self.assertAlmostEqual(standardized["value"].mean(), 0.0)
        self.assertEqual(df["value"].tolist(), [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            normalize(pd.DataFrame({"value": [1.0, np.nan]}), "value")

    def test_label_encoding_rejects_missing_and_does_not_mutate(self):
        df = pd.DataFrame({"label": ["cat", "dog", "cat"]})

        encoded = label_encode(df, "label")

        self.assertEqual(encoded["label"].tolist(), [0, 1, 0])
        self.assertEqual(df["label"].tolist(), ["cat", "dog", "cat"])
        with self.assertRaisesRegex(ValueError, "missing values"):
            label_encode(pd.DataFrame({"label": ["cat", None]}), "label")

    def test_one_hot_encoding_and_column_drop(self):
        df = pd.DataFrame({"kind": ["a", "b"], "value": [1, 2]})

        encoded = get_dummies(df, "kind")
        dropped = drop_selected_columns(df, "kind")

        self.assertTrue(any(name.startswith("kind_") for name in encoded.columns))
        self.assertEqual(list(dropped.columns), ["value"])
        self.assertEqual(list(df.columns), ["kind", "value"])
        with self.assertRaisesRegex(ValueError, "Unknown column"):
            drop_selected_columns(df, "typo")

    def test_duplicate_policies(self):
        df = pd.DataFrame({"value": [1, 1, 2]})

        self.assertEqual(len(manage_duplicates(df)), 2)
        self.assertEqual(len(manage_duplicates(df, action="drop_all")), 1)
        with self.assertRaisesRegex(ValueError, "Action must be"):
            manage_duplicates(df, action="unknown")


class LeakageSafePreprocessingTests(unittest.TestCase):
    def test_unselected_categorical_columns_are_excluded(self):
        X_train = pd.DataFrame({
            "amount": [1.0, 2.0, 3.0],
            "Sex": ["male", "female", "male"],
            "Ticket": ["A/1", "B/2", "C/3"],
        })
        X_test = pd.DataFrame({
            "amount": [4.0],
            "Sex": ["female"],
            "Ticket": ["D/4"],
        })

        train, test, metadata = preprocess_split_features(
            X_train,
            X_test,
            categorical_columns=["Sex"],
            categorical_encoding="One-hot",
        )

        self.assertEqual(
            train.columns.tolist(),
            [
                "numeric__amount",
                "categorical__Sex_female",
                "categorical__Sex_male",
            ],
        )
        self.assertEqual(test.columns.tolist(), train.columns.tolist())
        self.assertEqual(metadata["categorical_columns"], ["Sex"])
        self.assertEqual(metadata["excluded_categorical_columns"], ["Ticket"])

    def test_preprocessing_fits_imputation_encoding_scaling_and_caps_on_train(self):
        X_train = pd.DataFrame({
            "amount": [0.0, 1.0, 2.0, np.nan, 100.0],
            "kind": ["a", "b", "a", None, "b"],
        })
        X_test = pd.DataFrame({
            "amount": [1000.0, np.nan],
            "kind": ["unseen", None],
        })

        train, test, metadata = preprocess_split_features(
            X_train,
            X_test,
            scaling="Normalization",
            outlier_method="IQR",
            categorical_encoding="One-hot",
            categorical_columns=["kind"],
        )

        self.assertEqual(len(train), len(X_train))
        self.assertEqual(len(test), len(X_test))
        self.assertEqual(list(train.columns), list(test.columns))
        self.assertFalse(train.isna().any().any())
        self.assertFalse(test.isna().any().any())
        self.assertIn("amount", metadata["outlier_bounds"])
        self.assertLessEqual(test["numeric__amount"].max(), 1.0)
        self.assertTrue(any("kind_" in name for name in train.columns))
        self.assertEqual(metadata["scaling"], "Normalization")
        self.assertEqual(metadata["outlier_method"], "IQR")

    def test_configurable_imputation_is_fitted_only_on_training_features(self):
        X_train = pd.DataFrame({
            "amount": [1.0, np.nan, 3.0],
            "kind": ["a", None, "b"],
        })
        X_test = pd.DataFrame({
            "amount": [1000.0, np.nan],
            "kind": ["unseen", None],
        })

        train, test, metadata = preprocess_split_features(
            X_train,
            X_test,
            numeric_imputation="mean",
            categorical_imputation="constant",
            categorical_fill_value="unknown",
            categorical_encoding="One-hot",
            categorical_columns=["kind"],
        )

        self.assertEqual(train["numeric__amount"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(test["numeric__amount"].tolist(), [1000.0, 2.0])
        self.assertIn("categorical__kind_unknown", train.columns)
        self.assertEqual(metadata["numeric_imputation"], "mean")
        self.assertEqual(metadata["categorical_imputation"], "constant")

        empty_train, empty_test, _ = preprocess_split_features(
            pd.DataFrame({"empty_number": [np.nan, np.nan]}),
            pd.DataFrame({"empty_number": [7.0]}),
            numeric_imputation="constant",
            numeric_fill_value=0,
            categorical_encoding="One-hot",
        )
        self.assertEqual(empty_train["numeric__empty_number"].tolist(), [0.0, 0.0])
        self.assertEqual(empty_test["numeric__empty_number"].tolist(), [7.0])

    def test_unknown_categories_are_supported_and_test_does_not_affect_scaler(self):
        X_train = pd.DataFrame({
            "number": [0.0, 10.0, 20.0],
            "category": ["train-a", "train-b", "train-a"],
            "empty_category": pd.Series([None, None, None], dtype="string"),
        })
        X_test = pd.DataFrame({
            "number": [10000.0],
            "category": ["test-only"],
            "empty_category": pd.Series(["test-only"], dtype="string"),
        })

        train, test, metadata = preprocess_split_features(
            X_train,
            X_test,
            scaling="Standardization",
            categorical_encoding="One-hot",
            categorical_columns=["category", "empty_category"],
        )

        self.assertEqual(float(train["numeric__number"].mean()), 0.0)
        self.assertGreater(float(test["numeric__number"].iloc[0]), 100.0)
        self.assertEqual(len(train.columns), len(test.columns))
        self.assertTrue(np.isfinite(test.to_numpy()).all())
        self.assertEqual(metadata["all_missing_categorical_columns"], ["empty_category"])

    def test_ordinal_encoding_uses_training_categories_and_marks_unknown_test_values(self):
        X_train = pd.DataFrame({"kind": ["low", "high", "low", None]})
        X_test = pd.DataFrame({"kind": ["unseen", "high"]})

        train, test, metadata = preprocess_split_features(
            X_train,
            X_test,
            categorical_encoding="Ordinal",
            categorical_columns=["kind"],
        )

        self.assertEqual(train.columns.tolist(), ["categorical__kind"])
        self.assertEqual(test["categorical__kind"].tolist(), [-1.0, 0.0])
        self.assertEqual(metadata["categorical_encoding"], "Ordinal")

    def test_percentile_and_zscore_caps(self):
        train_data = pd.DataFrame({"number": list(range(1, 101))})
        test_data = pd.DataFrame({"number": [-1000.0, 1000.0]})

        _, percentile_test, percentile_meta = preprocess_split_features(
            train_data,
            test_data,
            outlier_method="Percentile",
            lower_percentile=10,
            upper_percentile=90,
            categorical_encoding="One-hot",
        )
        _, zscore_test, zscore_meta = preprocess_split_features(
            train_data,
            test_data,
            outlier_method="Z-score",
            categorical_encoding="One-hot",
        )

        self.assertEqual(percentile_meta["outlier_method"], "Percentile")
        self.assertEqual(zscore_meta["outlier_method"], "Z-score")
        percentile_bounds = percentile_meta["outlier_bounds"]["number"]
        zscore_bounds = zscore_meta["outlier_bounds"]["number"]
        self.assertGreaterEqual(
            percentile_test["numeric__number"].min(), percentile_bounds[0]
        )
        self.assertLessEqual(
            percentile_test["numeric__number"].max(), percentile_bounds[1]
        )
        self.assertGreaterEqual(
            zscore_test["numeric__number"].min(), zscore_bounds[0]
        )
        self.assertLessEqual(
            zscore_test["numeric__number"].max(), zscore_bounds[1]
        )

    def test_preprocessing_rejects_misaligned_or_invalid_configuration(self):
        train = pd.DataFrame({"x": [1, 2]})
        test = pd.DataFrame({"different": [3]})
        with self.assertRaisesRegex(ValueError, "identical columns"):
            preprocess_split_features(
                train,
                test,
                categorical_encoding="One-hot",
            )
        with self.assertRaisesRegex(ValueError, "Percentiles"):
            preprocess_split_features(
                train,
                train,
                outlier_method="Percentile",
                lower_percentile=90,
                upper_percentile=10,
                categorical_encoding="One-hot",
            )


class ColumnEditingTests(unittest.TestCase):
    def test_rename_column_validates_names_and_does_not_mutate(self):
        df = pd.DataFrame({"old": [1], "existing": [2]})

        renamed = rename_column(df, "old", "new")

        self.assertEqual(list(renamed.columns), ["new", "existing"])
        self.assertEqual(list(df.columns), ["old", "existing"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            rename_column(df, "old", "existing")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            rename_column(df, "missing", "new")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            rename_column(df, "old", " ")

    def test_change_dtype_converts_numeric_datetime_and_boolean_values(self):
        df = pd.DataFrame({
            "amount": ["1", "2.5", None],
            "date": ["2024-01-01", "2024-01-02", None],
            "active": ["yes", "no", None],
        })

        converted = change_column_dtype(df, "amount", "Float64")
        converted = change_column_dtype(converted, "date", "datetime64[ns]")
        converted = change_column_dtype(converted, "active", "boolean")

        self.assertEqual(str(converted["amount"].dtype), "Float64")
        self.assertTrue(pd.api.types.is_datetime64_dtype(converted["date"]))
        self.assertEqual(str(converted["active"].dtype), "boolean")
        self.assertEqual(converted["active"].tolist(), [True, False, pd.NA])
        self.assertEqual(df["amount"].iloc[:2].tolist(), ["1", "2.5"])
        self.assertTrue(pd.isna(df["amount"].iloc[2]))

    def test_change_dtype_rejects_bad_data_and_invalid_boolean_text(self):
        df = pd.DataFrame({"amount": ["1", "not-a-number"]})
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            change_column_dtype(df, "amount", "float64")

        flags = pd.DataFrame({"active": ["yes", "sometimes"]})
        with self.assertRaisesRegex(ValueError, "booleans"):
            change_column_dtype(flags, "active", "bool")

    def test_history_supports_undo_reset_and_atomic_failures(self):
        original = pd.DataFrame({"old": [1, 2]})
        history = DataPreparationHistory(original)

        history.apply(rename_column, "old", "new")
        history.apply(change_column_dtype, "new", "float64")
        self.assertEqual(str(history.current["new"].dtype), "float64")

        undone = history.undo()
        self.assertEqual(str(undone["new"].dtype), "int64")
        reset = history.reset()
        pd.testing.assert_frame_equal(reset, original)
        with self.assertRaisesRegex(ValueError, "no preparation step"):
            history.undo()

        def invalid_transform(_):
            raise RuntimeError("failed transform")

        with self.assertRaisesRegex(RuntimeError, "failed transform"):
            history.apply(invalid_transform)
        pd.testing.assert_frame_equal(history.current, original)


if __name__ == "__main__":
    unittest.main()
