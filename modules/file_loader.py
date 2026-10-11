from pathlib import Path

import pandas as pd


def load_file(path):
    file_path = Path(path)
    extension = file_path.suffix.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(file_path)

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

        print(f"Loaded successfully: {file_path.name}")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes")

        return df

    except Exception as e:
        raise ValueError(
            f"Failed to load file '{file_path}': {e}"
        ) from e
