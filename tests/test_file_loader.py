import tempfile
import unittest
from pathlib import Path

from modules.file_loader import load_file


class FileLoaderTests(unittest.TestCase):
    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("name,value\nAda,1\n", encoding="utf-8")
            df = load_file(path)
            self.assertEqual(list(df.columns), ["name", "value"])
            self.assertEqual(len(df), 1)

    def test_txt_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("hello world\nsecond line", encoding="utf-8")
            stats = load_file(path)
            self.assertEqual(stats["file_type"], "TXT")
            self.assertEqual(stats["word_count"], 4)
            self.assertEqual(stats["line_count"], 2)

    def test_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_file(path)

    def test_malformed_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_bytes(b"\x00\x01\x02not,csv")
            with self.assertRaisesRegex(ValueError, "null bytes"):
                load_file(path)

    def test_csv_rejects_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_encoding.csv"
            path.write_bytes(b"name,value\nAda,\xff\n")
            with self.assertRaisesRegex(ValueError, "Failed to load file"):
                load_file(path)

    def test_csv_rejects_rows_with_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_rows.csv"
            path.write_text("name,value\nAda,1,unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Failed to load file"):
                load_file(path)

    def test_missing_file_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.csv"
            with self.assertRaisesRegex(ValueError, "Failed to load file"):
                load_file(path)


if __name__ == "__main__":
    unittest.main()
