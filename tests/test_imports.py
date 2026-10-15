import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_shared_modules_import(self):
        for name in (
            "modules.ui",
            "modules.data_utils",
            "modules.filters",
            "modules.charts",
            "modules.file_loader",
            "modules.data_prepration",
        ):
            importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
