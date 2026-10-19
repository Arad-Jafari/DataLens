# DataLens

**Explore, understand, clean, manage, and prepare tabular data for machine
learning — all from an interactive Streamlit workspace.**

DataLens turns CSV and Excel files into a practical data workflow. Upload a
dataset, inspect its quality, explore distributions and relationships, edit
records, and prepare leakage-safe training and test features without changing
the active workspace until you choose to apply your changes.

## What DataLens can do

### Upload and inspect data

- Upload CSV and XLSX datasets; view basic statistics for TXT files.
- Preview records and summarize rows, columns, missing cells, memory use, and
  duplicate rows.
- Explore columns, data types, unique values, descriptive statistics, and
  missing-value reports.

### Understand and visualize

- Search and filter data with quick search and configurable advanced
  conditions.
- Visualize numeric distributions with histograms and box plots.
- Compare numeric variables with scatter plots and correlation heatmaps.
- Explore category counts and frequency distributions.
- Inspect potential outliers without assuming every flagged value is invalid.

### Manage records with CRUD

The **Dataset Management** page supports:

- **Create** records by entering values for the dataset columns.
- **Read** records in a paginated table.
- **Update** a selected record.
- **Delete** a selected record with a confirmation step.
- Search records by an `id` column when available, or by record number otherwise.
- Undo the last CRUD change or reset the working copy.
- Apply the edited dataset to Home and Data Explorer only when ready.

CRUD edits are made to a working copy first. They are session-scoped and are
not written back to the source CSV/XLSX file.

### Prepare data for machine learning

The **ML Preparation** workflow is organized into stages:

1. **Column changes** — rename columns, change data types, remove unwanted
   columns, or convert numeric-looking text into numbers.
2. **Data cleaning** — review dataset quality, remove rows with missing values
   when appropriate, and manage duplicate rows.
3. **Target and split** — choose supervised learning to select a target and
   create reproducible train/test partitions, or choose unsupervised clustering
   to proceed without a target or split.
4. **Model preprocessing** — for supervised work, fit transformations on
   training features and reuse them on test features. For clustering, fit and
   apply transformations to the full feature dataset.
5. **Review and apply** — inspect and download supervised splits or prepared
   clustering features, or explicitly apply the cleaned dataset to the workspace.

Model preprocessing includes:

- Configurable numeric median, mean, or constant-value imputation and
  categorical most-frequent or constant-value imputation.
- Explicit per-column selection of categorical features to encode, followed by
  a one-hot or ordinal choice. Unselected categorical columns are excluded from
  model features; encoders are fitted on training data and safely handle
  test-only categories.
- Optional numeric standardization or min-max normalization.
- Optional training-fitted outlier capping using IQR, percentile, or z-score
  thresholds. Thresholds come from `X_train`; values in both sets are capped
  without deleting test rows.
- For clustering, the same preparation choices are available without a target;
  transformations are fitted on the full dataset, and the processed feature
  matrix can be downloaded.
- An explicit conversion action for numeric values stored as text, such as
  `"12.5"`.

For supervised work, preprocessing statistics and thresholds are fitted on
training features only and reused on test features to avoid test-set leakage.
For clustering, there is no target or split; transformations are fitted on the
full dataset for exploratory unsupervised preparation.

The preparation workflow supports undo/reset before applying changes. Applying
a prepared dataset updates the active workspace and refreshes Data Explorer;
it does not overwrite the uploaded file.

## Quick start

### Requirements

- Python 3.12 or compatible
- Windows, macOS, or Linux

### Install and run

```bash
git clone https://github.com/Arad-Jafari/DataLens.git
cd DataLens
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your
browser.

## Typical workflow

1. Upload a CSV or XLSX file on **Home**.
2. Open **Data Explorer** to inspect data quality, distributions, and
   relationships.
3. Use **Dataset Management** for record-level CRUD changes, or **ML
   Preparation** for data transformations and train/test preparation.
4. Review changes and download outputs, or apply the working copy to the
   active workspace.

Changes are held in Streamlit session state while the session is active. Keep
downloads of important results; the app does not provide persistent database
storage or write changes back to the original uploaded file.

## Project structure

```text
.
├── app.py                         # Home page, upload flow, and dataset summary
├── modules/
│   ├── charts.py                  # Plotly chart builders
│   ├── crud.py                    # Validated DataFrame CRUD operations
│   ├── data_preparation.py        # Cleaning, splitting, and preprocessing
│   ├── data_utils.py              # Shared dataset summaries and helpers
│   ├── file_loader.py             # CSV, XLSX, and TXT loading
│   ├── filters.py                 # Search and filter logic
│   └── ui.py                      # Shared session state and navigation
├── pages/
│   ├── 1_Data_Explorer.py         # Data understanding and visual reports
│   ├── 2_Data_Preparation.py      # ML preparation workflow
│   └── 3_Data_CRUD.py             # Dataset record management
├── tests/                         # Unit tests
└── .github/workflows/tests.yml    # Continuous integration
```

## Current scope

DataLens focuses on data exploration, record management, and dataset
preparation. It creates train/test data and transforms features, but it does
not yet train or evaluate machine-learning models. Data is processed in memory
within the active Streamlit session; database persistence, authentication,
and multi-user collaboration are outside the current scope.
