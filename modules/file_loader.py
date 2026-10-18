import csv
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd


def load_file(path):
    """Load a supported file, reporting invalid input as a clear ValueError."""
    try:
        file_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid file path: {path!r}") from exc

    extension = file_path.suffix.lower()

    try:
        if extension == ".csv":
            csv_bytes = file_path.read_bytes()
            if b"\x00" in csv_bytes:
                raise ValueError("CSV files must not contain null bytes.")
            csv_text = csv_bytes.decode("utf-8-sig")
            expected_fields = None
            for row in csv.reader(StringIO(csv_text, newline=""), strict=True):
                if not row:
                    continue
                if expected_fields is None:
                    expected_fields = len(row)
                elif len(row) != expected_fields:
                    raise ValueError(
                        "CSV rows must contain the same number of fields."
                    )
            df = pd.read_csv(StringIO(csv_text), on_bad_lines="error")

        elif extension == ".xlsx":
            df = pd.read_excel(file_path)

        elif extension == ".txt":
            text = file_path.read_text(encoding="utf-8")
            file_size = file_path.stat().st_size

            return {
                "file_name": file_path.name,
                "file_type": "TXT",
                "file_size_bytes": file_size,
                "character_count": len(text),
                "line_count": len(text.splitlines()),
                "word_count": len(text.split()),
            }

        else:
            raise ValueError(
                f"Unsupported file format: {extension or 'unknown'}"
            )

        return df

    except (
        OSError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
        csv.Error,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ImportError,
    ) as e:
        raise ValueError(
            f"Failed to load file '{file_path}': {e}"
        ) from e
