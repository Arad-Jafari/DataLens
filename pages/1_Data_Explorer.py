import hashlib
from datetime import date
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataLens | Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #263449;
    }

    .section-label {
        color: #94A9C5;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background: #151F32;
        border: 1px solid #29374D;
        border-radius: 14px;
        padding: 16px;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 2px 16px;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        background: #1D3554;
        border: 1px solid #345477;
        border-radius: 12px;
        font-size: 21px;
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #F1F5F9;
    }

    .sidebar-subtitle {
        color: #8198B7;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: 1.4px;
    }

    .sidebar-section {
        color: #8198B7;
        font-size: .68rem;
        font-weight: 750;
        letter-spacing: 1.5px;
        margin: 12px 0 8px;
    }

    .sidebar-dataset {
        background: #121E30;
        border: 1px solid #293B54;
        border-radius: 12px;
        padding: 14px;
        margin: 8px 0 12px;
        overflow-wrap: anywhere;
    }

    .dataset-name {
        color: #F1F5F9;
        font-size: .9rem;
        font-weight: 650;
        margin-bottom: 5px;
    }

    .dataset-details {
        color: #91A5C0;
        font-size: .78rem;
        line-height: 1.6;
    }

    [data-testid="stSidebar"] hr,
    hr {
        border-color: #29374D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE HELPERS
# ============================================================

def clear_search_and_filters():
    """Clear search and filter controls without changing the source data."""
    st.session_state["dl_quick_search"] = ""
    st.session_state["dl_filter_logic"] = "AND"
    st.session_state["dl_filter_drafts"] = []
    st.session_state["dl_applied_conditions"] = []
    st.session_state["dl_applied_logic"] = "AND"
    st.session_state["dl_exclude_missing"] = False


def reset_search_and_filters():
    """Reset search, filters, and displayed columns."""
    clear_search_and_filters()

    source = st.session_state.get("working_df")
    if isinstance(source, pd.DataFrame):
        st.session_state["dl_visible_columns"] = list(source.columns)


def get_dataset_signature(df):
    """Generate a robust signature for ordinary CSV/Excel datasets."""
    try:
        safe = df.astype("string").fillna("<NA>")
        row_hashes = pd.util.hash_pandas_object(
            safe,
            index=True,
        ).values.tobytes()
        payload = row_hashes + repr(list(df.columns)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except (TypeError, ValueError):
        return hashlib.sha256(
            repr((df.shape, list(map(str, df.columns)), df.dtypes.astype(str).tolist()))
            .encode("utf-8")
        ).hexdigest()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🔎</div>
            <div>
                <div class="sidebar-title">DataLens</div>
                <div class="sidebar-subtitle">DATA INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<p class="sidebar-section">WORKSPACE</p>',
        unsafe_allow_html=True,
    )

    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link(
        "pages/1_Data_Explorer.py",
        label="Data Explorer",
        icon="📊",
    )

    st.markdown(
        '<p class="sidebar-section">DATASET</p>',
        unsafe_allow_html=True,
    )

    sidebar_df = st.session_state.get("working_df")
    sidebar_name = st.session_state.get(
        "dataset_name",
        "Current dataset",
    )

    if isinstance(sidebar_df, pd.DataFrame):
        st.markdown(
            f"""
            <div class="sidebar-dataset">
                <div class="dataset-name">{escape(str(sidebar_name))}</div>
                <div class="dataset-details">
                    {sidebar_df.shape[0]:,} rows<br>
                    {sidebar_df.shape[1]:,} columns
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a dataset on the Home page to begin.")

    st.divider()
    st.markdown(
        '<p class="sidebar-section">SUPPORTED FILES</p>',
        unsafe_allow_html=True,
    )
    st.markdown("`CSV`  `XLSX`")
    st.divider()
    st.caption("DataLens · Data made clearer.")


# ============================================================
# LOAD DATASET
# ============================================================

st.markdown("")
st.markdown("DATA WORKSPACE / EXPLORATION")
st.title("Data Explorer")
st.write(
    "Search, filter, visualize, and understand your dataset "
    "without changing the original data."
)

df = st.session_state.get("working_df")

if not isinstance(df, pd.DataFrame):
    st.info(
        "No valid dataset is loaded yet. Go to Home and upload "
        "a CSV or Excel file to start exploring."
    )
    st.stop()

dataset_name = st.session_state.get(
    "dataset_name",
    "Current dataset",
)

if df.empty:
    st.warning("Your dataset contains no rows.")

if df.columns.empty:
    st.warning("Your dataset contains no columns.")
    st.stop()


# ============================================================
# RESET SEARCH STATE WHEN DATASET CHANGES
# ============================================================

signature = get_dataset_signature(df)

if st.session_state.get("dl_dataset_signature") != signature:
    for key in list(st.session_state.keys()):
        if key.startswith("dl_"):
            del st.session_state[key]

    st.session_state["dl_dataset_signature"] = signature
    st.session_state["dl_quick_search"] = ""
    st.session_state["dl_filter_logic"] = "AND"
    st.session_state["dl_filter_drafts"] = []
    st.session_state["dl_applied_conditions"] = []
    st.session_state["dl_applied_logic"] = "AND"
    st.session_state["dl_exclude_missing"] = False
    st.session_state["dl_visible_columns"] = list(df.columns)


# ============================================================
# FILTER HELPERS
# ============================================================

def infer_column_type(series):
    """Infer numeric, date, or text type for a filter control."""
    if pd.api.types.is_bool_dtype(series):
        return "text"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    values = series.dropna()
    if values.empty:
        return "text"

    sample = values.astype(str).head(100)

    numeric = pd.to_numeric(
        sample.str.replace(",", "", regex=False),
        errors="coerce",
    )
    if numeric.notna().mean() >= 0.98:
        return "numeric"

    if sample.str.contains(r"[-/:T]", regex=True).mean() >= 0.8:
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return "date"

    return "text"


def parse_numeric(value):
    if value is None or not str(value).strip():
        return None

    try:
        result = float(str(value).replace(",", "").strip())
        if pd.isna(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def apply_one_condition(source, condition):
    """Return a boolean mask for one validated condition."""
    column = condition["column"]
    operator = condition["operator"]
    kind = condition["kind"]
    value = condition.get("value")
    series = source[column]

    if operator == "Is missing":
        return series.isna()

    if operator == "Is not missing":
        return series.notna()

    if kind == "text":
        text = series.astype("string")

        if operator == "Contains":
            return text.str.contains(
                str(value),
                case=True,
                regex=False,
                na=False,
            )
        if operator == "Equals":
            return text.eq(str(value)).fillna(False)
        if operator == "Not equal to":
            return text.ne(str(value)).fillna(False)
        if operator == "Starts with":
            return text.str.startswith(str(value), na=False)
        if operator == "Ends with":
            return text.str.endswith(str(value), na=False)

    elif kind == "numeric":
        numeric = pd.to_numeric(
            series.astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )

        if operator == "Between":
            result = numeric.between(value[0], value[1], inclusive="both")
        elif operator == "Greater than":
            result = numeric > value
        elif operator == "Greater than or equal to":
            result = numeric >= value
        elif operator == "Less than":
            result = numeric < value
        elif operator == "Less than or equal to":
            result = numeric <= value
        elif operator == "Equals":
            result = numeric == value
        elif operator == "Not equal to":
            result = numeric != value
        else:
            raise ValueError(f"Unsupported numeric operator: {operator}")

        return result & numeric.notna()

    elif kind == "date":
        dates = pd.to_datetime(series, errors="coerce")

        if operator == "Between":
            result = dates.between(
                pd.Timestamp(value[0]),
                pd.Timestamp(value[1]),
                inclusive="both",
            )
        elif operator == "Before":
            result = dates < pd.Timestamp(value)
        elif operator == "After":
            result = dates > pd.Timestamp(value)
        elif operator == "On":
            result = dates.dt.date == value
        else:
            raise ValueError(f"Unsupported date operator: {operator}")

        return result & dates.notna()

    raise ValueError(f"Unsupported filter condition: {kind} / {operator}")


def build_filtered_view(source, conditions, logic="AND"):
    """Apply filters positionally so duplicate DataFrame indices are safe."""
    if not conditions:
        return source.copy()

    combined = None

    for condition in conditions:
        mask = apply_one_condition(source, condition)
        mask = pd.Series(mask, index=source.index).fillna(False)

        if combined is None:
            combined = mask
        elif logic == "OR":
            combined = combined | mask
        else:
            combined = combined & mask

    return source.loc[combined].copy()


def make_frequency_table(series, grouped=False, bins=10, limit=100):
    values = series.dropna()

    if values.empty:
        return pd.DataFrame(
            columns=[
                "Value / Interval",
                "Frequency",
                "Relative Frequency (%)",
                "Cumulative Frequency",
                "Cumulative Percentage (%)",
            ]
        )

    if grouped and pd.api.types.is_numeric_dtype(values):
        counts = pd.cut(
            values,
            bins=bins,
            include_lowest=True,
        ).value_counts(sort=False)
    else:
        counts = values.value_counts(dropna=False)
        if len(counts) > limit:
            counts = counts.head(limit)

    table = counts.rename_axis("Value / Interval").reset_index(name="Frequency")
    table["Value / Interval"] = table["Value / Interval"].astype(str)
    total = int(table["Frequency"].sum())

    table["Relative Frequency (%)"] = (
        table["Frequency"] / total * 100
    ).round(2)

    table["Cumulative Frequency"] = table["Frequency"].cumsum()

    table["Cumulative Percentage (%)"] = (
        table["Cumulative Frequency"] / total * 100
    ).round(2)

    return table


# ============================================================
# TABS
# ============================================================

search_tab, quality_tab, columns_tab, stats_tab, reports_tab = st.tabs(
    [
        "🔎 Search & Filter",
        "🩺 Data Quality",
        "📋 Columns",
        "📈 Statistics",
        "📊 Visual Reports",
    ]
)


# ============================================================
# TAB 1 — SEARCH AND FILTER
# ============================================================

with search_tab:
    st.subheader("Search & Filter")
    st.caption(
        "Search is case-sensitive. Advanced conditions combine with "
        "the selected AND/OR rule. The original dataset is never modified."
    )

    quick_search = st.text_input(
        "Quick search",
        key="dl_quick_search",
        placeholder="Enter text to search...",
    )

    search_scope = st.radio(
        "Search scope",
        ["Entire dataset", "Selected column"],
        horizontal=True,
        key="dl_search_scope",
    )

    search_column = None

    if search_scope == "Selected column":
        search_column = st.selectbox(
            "Column to search",
            options=list(df.columns),
            key="dl_search_column",
        )

    st.checkbox(
        "Exclude rows with missing values",
        key="dl_exclude_missing",
        help=(
            "When enabled, remove rows containing a missing value "
            "in any column from Search Results and its chart only."
        ),
    )

    with st.expander("Advanced filters", expanded=True):
        filter_logic = st.radio(
            "Combine advanced conditions",
            ["AND", "OR"],
            horizontal=True,
            key="dl_filter_logic",
        )

        drafts = st.session_state.setdefault("dl_filter_drafts", [])

        if st.button("＋ Add condition", key="dl_add_condition"):
            drafts.append({
                "id": max(
                    [item["id"] for item in drafts],
                    default=0,
                ) + 1,
            })
            st.rerun()

        remove_positions = []

        for position, draft in enumerate(drafts):
            condition_id = draft["id"]

            st.markdown(f"**Condition {position + 1}**")

            selected_column = st.selectbox(
                "Filter column",
                options=list(df.columns),
                key=f"dl_col_{condition_id}",
            )

            detected_type = infer_column_type(df[selected_column])

            if detected_type == "text" and df[selected_column].dtype == object:
                selected_kind_label = st.selectbox(
                    "Interpret column as",
                    ["Text", "Numeric", "Date/time"],
                    key=f"dl_kind_{condition_id}",
                )
                kind = {
                    "Text": "text",
                    "Numeric": "numeric",
                    "Date/time": "date",
                }[selected_kind_label]
            else:
                kind = detected_type
                st.caption(f"Detected type: {kind}")

            operators = {
                "text": [
                    "Contains",
                    "Equals",
                    "Not equal to",
                    "Starts with",
                    "Ends with",
                    "Is missing",
                    "Is not missing",
                ],
                "numeric": [
                    "Between",
                    "Greater than",
                    "Greater than or equal to",
                    "Less than",
                    "Less than or equal to",
                    "Equals",
                    "Not equal to",
                    "Is missing",
                    "Is not missing",
                ],
                "date": [
                    "Between",
                    "Before",
                    "After",
                    "On",
                    "Is missing",
                    "Is not missing",
                ],
            }[kind]

            operator = st.selectbox(
                "Operator",
                operators,
                key=f"dl_op_{condition_id}",
            )

            value = None

            if operator not in ("Is missing", "Is not missing"):
                if kind == "text":
                    value = st.text_input(
                        "Text value",
                        key=f"dl_val_{condition_id}",
                    )

                elif kind == "numeric":
                    if operator == "Between":
                        low, high = st.columns(2)
                        with low:
                            lower = st.text_input(
                                "Minimum",
                                key=f"dl_low_{condition_id}",
                            )
                        with high:
                            upper = st.text_input(
                                "Maximum",
                                key=f"dl_high_{condition_id}",
                            )
                        value = [parse_numeric(lower), parse_numeric(upper)]
                    else:
                        raw = st.text_input(
                            "Numeric value",
                            key=f"dl_val_{condition_id}",
                        )
                        value = parse_numeric(raw)

                elif kind == "date":
                    if operator == "Between":
                        low, high = st.columns(2)
                        with low:
                            lower_date = st.date_input(
                                "Start date",
                                value=date.today(),
                                key=f"dl_date_low_{condition_id}",
                            )
                        with high:
                            upper_date = st.date_input(
                                "End date",
                                value=date.today(),
                                key=f"dl_date_high_{condition_id}",
                            )
                        value = [lower_date, upper_date]
                    else:
                        value = st.date_input(
                            "Date",
                            value=date.today(),
                            key=f"dl_date_{condition_id}",
                        )

            draft.update({
                "column": selected_column,
                "kind": kind,
                "operator": operator,
                "value": value,
            })

            if st.button(
                "Remove condition",
                key=f"dl_remove_{condition_id}",
            ):
                remove_positions.append(position)

            st.divider()

        for position in reversed(remove_positions):
            drafts.pop(position)

        if remove_positions:
            st.rerun()

        apply_clicked = st.button(
            "Apply Filters",
            type="primary",
            key="dl_apply_filters",
        )

        if apply_clicked:
            validated = []
            errors = []

            for position, draft in enumerate(drafts, start=1):
                column = draft.get("column")
                kind = draft.get("kind")
                operator = draft.get("operator")
                value = draft.get("value")

                if column not in df.columns or not kind or not operator:
                    errors.append(f"Condition {position}: select a column and operator.")
                    continue

                if operator in ("Is missing", "Is not missing"):
                    validated.append({
                        "column": column,
                        "kind": kind,
                        "operator": operator,
                        "value": None,
                    })
                    continue

                if kind == "text":
                    if value is None or not str(value).strip():
                        errors.append(f"Condition {position}: enter a text value.")
                        continue

                elif kind == "numeric":
                    if operator == "Between":
                        if value[0] is None or value[1] is None:
                            errors.append(f"Condition {position}: enter both numeric bounds.")
                            continue
                        if value[0] > value[1]:
                            errors.append(f"Condition {position}: minimum exceeds maximum.")
                            continue
                    elif value is None:
                        errors.append(f"Condition {position}: enter a valid numeric value.")
                        continue

                elif kind == "date":
                    if operator == "Between" and value[0] > value[1]:
                        errors.append(f"Condition {position}: start date exceeds end date.")
                        continue

                validated.append({
                    "column": column,
                    "kind": kind,
                    "operator": operator,
                    "value": value,
                })

            if errors:
                for message in errors:
                    st.error(message)
            else:
                try:
                    build_filtered_view(df, validated, filter_logic)
                    st.session_state["dl_applied_conditions"] = validated
                    st.session_state["dl_applied_logic"] = filter_logic
                    st.success("Advanced filters applied.")
                except Exception as error:
                    st.error(f"Unable to apply filters: {error}")

        clear_col, reset_col = st.columns(2)

        with clear_col:
            st.button(
                "Clear All",
                key="dl_clear_all",
                on_click=clear_search_and_filters,
                use_container_width=True,
            )

        with reset_col:
            st.button(
                "Reset",
                key="dl_reset",
                on_click=reset_search_and_filters,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # CREATE SEARCH RESULT DATAFRAME
    # --------------------------------------------------------

    # Use positional indexes inside Search & Filter so duplicate source
    # indexes cannot cause rows to be selected more than once.
    search_source = df.reset_index(drop=True)
    result_mask = pd.Series(True, index=search_source.index)

    if quick_search:
        if search_scope == "Selected column" and search_column is not None:
            search_columns = [search_column]
        else:
            search_columns = list(search_source.columns)

        quick_mask = pd.Series(False, index=search_source.index)
        for column in search_columns:
            text = search_source[column].astype("string")
            quick_mask |= text.str.contains(
                quick_search,
                case=True,
                regex=False,
                na=False,
            )
        result_mask &= quick_mask

    applied_conditions = st.session_state.get("dl_applied_conditions", [])
    if applied_conditions:
        try:
            advanced_source = build_filtered_view(
                search_source,
                applied_conditions,
                st.session_state.get("dl_applied_logic", "AND"),
            )
            advanced_positions = set(advanced_source.index)
            advanced_mask = pd.Series(
                search_source.index.isin(advanced_positions),
                index=search_source.index,
            )
            result_mask &= advanced_mask
        except Exception as error:
            st.error(f"Could not calculate advanced filters: {error}")

    searched_df = search_source.loc[result_mask].copy()

    # Missing-value exclusion applies only to Search & Filter results.
    search_result_df = searched_df.copy()
    if st.session_state.get("dl_exclude_missing", False):
        search_result_df = search_result_df.dropna(axis=0, how="any").copy()

    st.divider()
    st.subheader("Search Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Original rows", f"{len(df):,}")
    m2.metric("Matching rows", f"{len(searched_df):,}")
    m3.metric("Rows shown", f"{len(search_result_df):,}")

    visible_columns = st.multiselect(
        "Columns to display",
        options=list(df.columns),
        default=st.session_state.get("dl_visible_columns", list(df.columns)),
        key="dl_visible_columns",
    )

    display_df = search_result_df.loc[:, visible_columns]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download search results (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="datalens_search_results.csv",
        mime="text/csv",
        key="dl_download_filtered",
    )

    st.caption(
        "The search results, missing-value option, and chart below do not change "
        "the original dataset used by the other tabs."
    )

    # --------------------------------------------------------
    # SEARCH-BASED VISUALIZATION
    # --------------------------------------------------------

    st.divider()
    st.subheader("Visualize Search Results")

    chart_df = search_result_df

    if chart_df.empty:
        st.info("No matching rows are available to visualize.")
    else:
        chart_type = st.selectbox(
            "Visualization type",
            [
                "Histogram",
                "Bar chart",
                "Box plot",
                "Scatter plot",
                "Correlation heatmap",
            ],
            key="dl_search_chart_type",
        )

        numeric_columns = list(chart_df.select_dtypes(include="number").columns)
        categorical_columns = [
            col for col in chart_df.columns
            if not pd.api.types.is_numeric_dtype(chart_df[col])
            and not pd.api.types.is_datetime64_any_dtype(chart_df[col])
        ]

        if chart_type == "Histogram":
            if numeric_columns:
                column = st.selectbox(
                    "Numeric column",
                    numeric_columns,
                    key="dl_search_hist_column",
                )
                values = pd.to_numeric(chart_df[column], errors="coerce").dropna()
                if not values.empty:
                    fig = px.histogram(
                        chart_df.assign(**{column: pd.to_numeric(chart_df[column], errors="coerce")}),
                        x=column,
                        template="plotly_dark",
                        title=f"Distribution of {column} in search results",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("The selected column has no numeric values to chart.")
            else:
                st.info("No numeric columns are available for a histogram.")

        elif chart_type == "Bar chart":
            if categorical_columns:
                column = st.selectbox(
                    "Category column",
                    categorical_columns,
                    key="dl_search_bar_column",
                )
                limit = st.slider(
                    "Number of categories",
                    5,
                    50,
                    20,
                    key="dl_search_bar_limit",
                )
                counts = (
                    chart_df[column]
                    .astype("string")
                    .fillna("(Missing)")
                    .value_counts()
                    .head(limit)
                    .rename_axis("Category")
                    .reset_index(name="Count")
                )
                fig = px.bar(
                    counts,
                    x="Category",
                    y="Count",
                    template="plotly_dark",
                    title=f"Category counts for {column}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns are available for a bar chart.")

        elif chart_type == "Box plot":
            if numeric_columns:
                column = st.selectbox(
                    "Numeric column",
                    numeric_columns,
                    key="dl_search_box_column",
                )
                values = pd.to_numeric(chart_df[column], errors="coerce").dropna()
                if not values.empty:
                    fig = px.box(
                        chart_df.assign(**{column: pd.to_numeric(chart_df[column], errors="coerce")}),
                        y=column,
                        template="plotly_dark",
                        title=f"Box plot of {column}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("The selected column has no numeric values to chart.")
            else:
                st.info("No numeric columns are available for a box plot.")

        elif chart_type == "Scatter plot":
            if len(numeric_columns) >= 2:
                x_col, y_col = st.columns(2)
                with x_col:
                    x = st.selectbox(
                        "X-axis",
                        numeric_columns,
                        key="dl_search_scatter_x",
                    )
                with y_col:
                    y_options = [col for col in numeric_columns if col != x]
                    y = st.selectbox(
                        "Y-axis",
                        y_options,
                        key="dl_search_scatter_y",
                    )

                color_col = None
                if categorical_columns:
                    color_col = st.selectbox(
                        "Color by (optional)",
                        ["None"] + categorical_columns,
                        key="dl_search_scatter_color",
                    )
                    if color_col == "None":
                        color_col = None

                plot_df = chart_df.copy()
                plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
                plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
                plot_df = plot_df.dropna(subset=[x, y])
                if not plot_df.empty:
                    fig = px.scatter(
                        plot_df,
                        x=x,
                        y=y,
                        color=color_col,
                        template="plotly_dark",
                        title=f"{y} vs {x} in search results",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No complete X/Y pairs are available for this scatter plot.")
            else:
                st.info("At least two numeric columns are required.")

        elif chart_type == "Correlation heatmap":
            if len(numeric_columns) >= 2:
                numeric_data = chart_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
                corr = numeric_data.corr()
                valid_columns = corr.columns[corr.notna().any()]
                corr = corr.loc[valid_columns, valid_columns]
                if corr.shape[0] >= 2:
                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu",
                        template="plotly_dark",
                        title="Search-result correlation heatmap",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough valid numeric correlations for a heatmap.")
            else:
                st.info("At least two numeric columns are required.")



# ============================================================
# COMMON REPORT DATA
# ============================================================

# All tabs outside Search & Filter always use the original dataset.
# Search filters and missing-value exclusion never alter these reports.
report_df = df.copy()

row_count, column_count = report_df.shape
missing_count = int(report_df.isna().sum().sum())
duplicate_count = int(report_df.duplicated().sum())
total_cells = row_count * column_count

completeness = (
    (total_cells - missing_count) / total_cells * 100
    if total_cells
    else 100.0
)

memory_kb = report_df.memory_usage(deep=True).sum() / 1024
numeric_df = report_df.select_dtypes(include="number")


# ============================================================
# TAB 2 — DATA QUALITY
# ============================================================

with quality_tab:
    st.subheader("Data Quality Report")
    st.caption("These metrics reflect the current search results.")

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{row_count:,}")
    b.metric("Columns", f"{column_count:,}")
    c.metric("Missing cells", f"{missing_count:,}")
    d.metric("Duplicate rows", f"{duplicate_count:,}")

    st.metric("Completeness", f"{completeness:.2f}%")
    st.progress(completeness / 100)

    st.markdown("#### Missing Values by Column")

    missing = report_df.isna().sum()
    missing_percent = (
        missing.astype(float).div(row_count).mul(100).round(2)
        if row_count
        else pd.Series(0.0, index=report_df.columns, dtype=float)
    )
    missing_report = pd.DataFrame({
        "Column": missing.index.astype(str),
        "Missing Values": missing.values,
        "Missing (%)": missing_percent.values,
        "Non-Missing Values": report_df.notna().sum().values,
    })

    st.dataframe(
        missing_report.sort_values("Missing Values", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if missing_count == 0:
        st.success("No missing values in the current result.")
    else:
        st.warning("The current result contains missing values.")

    st.markdown("#### Duplicate Rows")
    duplicate_rate = duplicate_count / row_count * 100 if row_count else 0
    st.metric("Duplicate row rate", f"{duplicate_rate:.2f}%")



# ============================================================
# TAB 3 — COLUMN INFORMATION
# ============================================================

with columns_tab:
    st.subheader("Column Information")
    st.caption("Column counts reflect the current search result.")

    column_report = pd.DataFrame({
        "Column": report_df.columns.astype(str),
        "Data Type": [str(dtype) for dtype in report_df.dtypes],
        "Total Rows": [len(report_df)] * len(report_df.columns),
        "Unique Values": report_df.nunique(dropna=True).values,
    })

    st.dataframe(
        column_report,
        use_container_width=True,
        hide_index=True,
    )

    if len(report_df.columns):
        selected_column = st.selectbox(
            "Inspect a column",
            list(report_df.columns),
            key="dl_inspect_column",
        )

        series = report_df[selected_column]

        common = (
            series.value_counts(dropna=False)
            .head(10)
            .rename_axis("Value")
            .reset_index(name="Frequency")
        )

        st.markdown("#### Most Common Values")
        st.dataframe(common, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 — STATISTICS
# ============================================================

with stats_tab:
    st.subheader("Descriptive Statistics")
    st.caption(
        "Descriptive statistics use the original dataset. Missing-value "
        "details are consolidated in the Data Quality tab."
    )

    st.markdown(
        "Missing-value counts and column metadata are shown in their dedicated tabs."
    )

    st.divider()
    st.markdown("#### Numeric Descriptive Statistics")

    if not numeric_df.empty:
        numeric_stats = numeric_df.describe().T
        st.dataframe(numeric_stats, use_container_width=True)
    else:
        st.info("No numeric columns are available.")

    st.divider()
    st.markdown("#### Categorical Descriptive Statistics")

    categorical_df = report_df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    if not categorical_df.empty:
        categorical_stats = categorical_df.describe().T
        st.dataframe(categorical_stats, use_container_width=True)
    else:
        st.info("No categorical columns are available.")


# ============================================================
# TAB 5 — VISUAL REPORTS
# ============================================================

with reports_tab:
    st.subheader("Visual Reports")
    st.caption(
        "These charts use the original dataset. Missing-value details are "
        "available in the Data Quality tab."
    )

    if report_df.empty:
        st.info("No rows are available for visualization.")
    else:


        # Histogram
        st.markdown("#### 2. Numeric Distribution")

        numeric_columns = list(
            report_df.select_dtypes(include="number").columns
        )

        if numeric_columns:
            selected_numeric = st.selectbox(
                "Numeric column",
                numeric_columns,
                key="dl_report_hist_column",
            )

            values = report_df[selected_numeric].dropna()

            if not values.empty:
                fig = px.histogram(
                    report_df,
                    x=selected_numeric,
                    nbins=20,
                    template="plotly_dark",
                    title=f"Distribution of {selected_numeric}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("The selected column contains no numeric values.")
        else:
            st.info("No numeric columns are available.")

        st.divider()

        # Category frequency
        st.markdown("#### 3. Categorical Value Frequencies")

        categorical_columns = [
            col for col in report_df.columns
            if not pd.api.types.is_numeric_dtype(report_df[col])
            and not pd.api.types.is_datetime64_any_dtype(report_df[col])
        ]

        if categorical_columns:
            selected_category = st.selectbox(
                "Categorical column",
                categorical_columns,
                key="dl_report_category_column",
            )

            category_counts = (
                report_df[selected_category]
                .astype("string")
                .fillna("(Missing)")
                .value_counts()
                .head(20)
                .rename_axis("Value")
                .reset_index(name="Frequency")
            )

            fig = px.bar(
                category_counts,
                x="Frequency",
                y="Value",
                orientation="h",
                template="plotly_dark",
                title=f"Most common values in {selected_category}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns are available.")

        st.divider()

        # Frequency table
        st.markdown("#### 4. Frequency Distribution")

        frequency_column = st.selectbox(
            "Column for frequency analysis",
            list(report_df.columns),
            key="dl_frequency_column",
        )

        frequency_series = report_df[frequency_column]
        grouped = False

        if pd.api.types.is_numeric_dtype(frequency_series):
            grouped = st.radio(
                "Numeric frequency method",
                ["Grouped intervals", "Exact values"],
                horizontal=True,
                key="dl_frequency_method",
            ) == "Grouped intervals"

        bins = 10
        if grouped:
            bins = st.slider(
                "Number of intervals",
                min_value=5,
                max_value=30,
                value=10,
                key="dl_frequency_bins",
            )

        frequency_table = make_frequency_table(
            frequency_series,
            grouped=grouped,
            bins=bins,
        )

        if not frequency_table.empty:
            fig = px.bar(
                frequency_table.head(30),
                x="Value / Interval",
                y="Frequency",
                template="plotly_dark",
                title=f"Frequency distribution of {frequency_column}",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                frequency_table,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download frequency table",
                data=frequency_table.to_csv(index=False).encode("utf-8"),
                file_name="datalens_frequency_distribution.csv",
                mime="text/csv",
                key="dl_download_frequency",
            )
        else:
            st.info("No values are available for frequency analysis.")

        st.divider()

        # Correlation heatmap
        st.markdown("#### 5. Numeric Correlation Heatmap")

        if len(numeric_columns) >= 2:
            correlation = report_df[numeric_columns].corr()
            valid_columns = correlation.columns[
                correlation.notna().any()
            ]
            correlation = correlation.loc[valid_columns, valid_columns]

            if correlation.shape[0] >= 2:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns.astype(str),
                        y=correlation.index.astype(str),
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1,
                        zmid=0,
                        colorbar_title="Correlation",
                        text=[
                            [
                                f"{value:.2f}" if pd.notna(value) else "N/A"
                                for value in row
                            ]
                            for row in correlation.values
                        ],
                        texttemplate="%{text}",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Correlation between numeric columns",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Correlation ranges from -1 to +1. "
                    "Correlation does not necessarily imply causation."
                )
            else:
                st.info("Not enough valid numeric correlations for a heatmap.")
        else:
            st.info("At least two numeric columns are required for a heatmap.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("DataLens · Data made clearer.")
