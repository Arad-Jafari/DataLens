# DataLens

DataLens is a Streamlit app for exploring tabular datasets, checking data
quality, visualizing patterns, and preparing train/test splits for machine
learning.

## Requirements

- Python 3.12

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV or XLSX file from the home page. TXT files are supported for
basic text statistics.

## Run tests

The test suite uses Python's built-in `unittest` runner:

```powershell
python -m unittest discover -s tests -v
```

## Project structure

- `app.py`: upload flow and dataset overview
- `pages/`: data exploration and ML preparation screens
- `modules/`: file loading, data utilities, filtering, charts, and shared UI
- `tests/`: unit tests for data handling and core functionality
