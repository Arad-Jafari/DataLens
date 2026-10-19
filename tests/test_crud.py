import unittest

import pandas as pd

from modules.crud import (
    create_record,
    delete_record,
    read_record,
    update_record,
)


class CrudOperationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "name": ["Ada", "Linus", "Grace"],
                "age": pd.Series([36, 45, 40], dtype="int64").array,
                "active": [True, False, True],
            },
            index=[5, 5, 8],
        )

    def test_create_record_appends_in_column_order_without_mutating_source(self):
        created = create_record(
            self.data,
            {"name": "Katherine", "age": 30, "active": True},
        )

        self.assertEqual(list(created.columns), list(self.data.columns))
        self.assertEqual(created.iloc[-1].to_dict(), {
            "name": "Katherine",
            "age": 30,
            "active": True,
        })
        self.assertEqual(len(self.data), 3)

    def test_read_and_update_use_position_not_duplicate_index_label(self):
        record = read_record(self.data, 1)
        updated = update_record(self.data, 1, {"name": "Updated"})

        self.assertEqual(record["name"], "Linus")
        self.assertEqual(updated.iloc[1]["name"], "Updated")
        self.assertEqual(updated.iloc[0]["name"], "Ada")
        self.assertEqual(self.data.iloc[1]["name"], "Linus")

    def test_update_allows_missing_integer_and_category_values(self):
        category_data = self.data.copy()
        category_data["level"] = pd.Categorical(["a", "b", "a"])

        updated = update_record(
            category_data,
            0,
            {"age": pd.NA, "level": "new"},
        )

        self.assertTrue(pd.isna(updated.iloc[0]["age"]))
        self.assertEqual(str(updated["age"].dtype), "Int64")
        self.assertEqual(updated.iloc[0]["level"], "new")
        self.assertIn("new", updated["level"].cat.categories)

    def test_delete_one_of_duplicate_index_rows(self):
        deleted = delete_record(self.data, 1)

        self.assertEqual(deleted["name"].tolist(), ["Ada", "Grace"])
        self.assertEqual(deleted.index.tolist(), [5, 8])
        self.assertEqual(len(self.data), 3)

    def test_invalid_records_positions_and_types_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Missing value"):
            create_record(self.data, {"name": "Only one"})
        with self.assertRaisesRegex(ValueError, "Unknown column"):
            update_record(self.data, 0, {"unknown": 1})
        with self.assertRaisesRegex(ValueError, "between 0"):
            read_record(self.data, 3)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            update_record(self.data, 0, {"age": "not numeric"})


if __name__ == "__main__":
    unittest.main()
