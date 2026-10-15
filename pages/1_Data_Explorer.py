from datetime import date

import pandas as pd
import streamlit as st

from modules.charts import (
    build_box_plot,
    build_category_bar,
    build_correlation_heatmap,
    build_correlation_heatmap_annotated,
    build_frequency_bar,
    build_histogram,
    build_scatter_plot,
)
from modules.data_utils import (
    categorical_columns,
    get_column_series,
    get_dataset_summary,
    make_frequency_table,
    numeric_columns,
)
from modules.filters import (
    OPERATORS_BY_KIND,
    combined_filter_mask,
    infer_column_type,
    parse_numeric,
    quick_search_mask,
    validate_filter_drafts,
)
from modules.ui import (
    clear_search_and_filters,
    current_dataset,
    ensure_app_session,
    initialize_explorer_state,
    inject_base_css,
    render_sidebar,
    reset_search_and_filters,
)

st.set_page_config(
    page_title="DataLens | Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_app_session()
inject_base_css()
render_sidebar()

st.markdown("")
st.markdown("DATA WORKSPACE / EXPLORATION")
st.title("Data Explorer")
st.write(
    "Search, filter, visualize, and understand your dataset "
    "without changing the original data."
)

df = current_dataset()

if not isinstance(df, pd.DataFrame):
    st.info(
        "No valid dataset is loaded yet. Go to Home and upload "
        "a CSV or Excel file to start exploring."
    )
    st.stop()

dataset_name = st.session_state.get("dataset_name") or "Current dataset"
dataset_id = st.session_state.get("processed_upload_id")
if dataset_id is None:
    dataset_id = (dataset_name, df.shape)

if df.empty:
    st.warning("Your dataset contains no rows.")

if df.columns.empty:
    st.warning("Your dataset contains no columns.")
    st.stop()

initialize_explorer_state(df, dataset_id)


search_tab, quality_tab, columns_tab, stats_tab, reports_tab = st.tabs(
    [
        "🔎 Search & Filter",
        "🩺 Data Quality",
        "📋 Columns",
        "📈 Statistics",
        "📊 Visual Reports",
    ]
)


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
                "id": max([item["id"] for item in drafts], default=0) + 1,
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

            detected_type = infer_column_type(get_column_series(df, selected_column))

            if detected_type == "text" and get_column_series(df, selected_column).dtype == object:
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

            operator = st.selectbox(
                "Operator",
                OPERATORS_BY_KIND[kind],
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

            if st.button("Remove condition", key=f"dl_remove_{condition_id}"):
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
            validated, errors = validate_filter_drafts(drafts, df.columns)
            if errors:
                for message in errors:
                    st.error(message)
            else:
                st.session_state["dl_applied_conditions"] = validated
                st.session_state["dl_applied_logic"] = filter_logic
                st.success("Advanced filters applied.")

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

    result_mask = pd.Series(True, index=range(len(df)), dtype=bool)

    if quick_search:
        if search_scope == "Selected column" and search_column is not None:
            search_columns = [search_column]
        else:
            search_columns = list(df.columns)
        search_hits = quick_search_mask(df, quick_search, search_columns)
        result_mask &= pd.Series(
            search_hits.to_numpy(),
            index=result_mask.index,
            dtype=bool,
        )

    applied_conditions = st.session_state.get("dl_applied_conditions", [])
    if applied_conditions:
        try:
            advanced_mask = combined_filter_mask(
                df,
                applied_conditions,
                st.session_state.get("dl_applied_logic", "AND"),
            )
            result_mask &= pd.Series(
                advanced_mask.to_numpy(),
                index=result_mask.index,
                dtype=bool,
            )
        except (TypeError, ValueError, KeyError) as error:
            st.error(f"Could not calculate advanced filters: {error}")

    searched_df = df.iloc[result_mask.to_numpy()].copy()

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

        chart_numeric = numeric_columns(chart_df)
        chart_categorical = categorical_columns(chart_df)

        if chart_type == "Histogram":
            if chart_numeric:
                column = st.selectbox(
                    "Numeric column",
                    chart_numeric,
                    key="dl_search_hist_column",
                )
                fig, message = build_histogram(
                    chart_df,
                    column,
                    f"Distribution of {column} in search results",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(message)
            else:
                st.info("No numeric columns are available for a histogram.")

        elif chart_type == "Bar chart":
            if chart_categorical:
                column = st.selectbox(
                    "Category column",
                    chart_categorical,
                    key="dl_search_bar_column",
                )
                limit = st.slider(
                    "Number of categories",
                    5,
                    50,
                    20,
                    key="dl_search_bar_limit",
                )
                fig, message = build_category_bar(
                    chart_df,
                    column,
                    f"Category counts for {column}",
                    limit=limit,
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(message)
            else:
                st.info("No categorical columns are available for a bar chart.")

        elif chart_type == "Box plot":
            if chart_numeric:
                column = st.selectbox(
                    "Numeric column",
                    chart_numeric,
                    key="dl_search_box_column",
                )
                fig, message = build_box_plot(
                    chart_df,
                    column,
                    f"Box plot of {column}",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(message)
            else:
                st.info("No numeric columns are available for a box plot.")

        elif chart_type == "Scatter plot":
            if len(chart_numeric) >= 2:
                x_col, y_col = st.columns(2)
                with x_col:
                    x = st.selectbox(
                        "X-axis",
                        chart_numeric,
                        key="dl_search_scatter_x",
                    )
                with y_col:
                    y_options = [col for col in chart_numeric if col != x]
                    y = st.selectbox(
                        "Y-axis",
                        y_options,
                        key="dl_search_scatter_y",
                    )
                color_col = None
                if chart_categorical:
                    color_col = st.selectbox(
                        "Color by (optional)",
                        ["None"] + chart_categorical,
                        key="dl_search_scatter_color",
                    )
                    if color_col == "None":
                        color_col = None
                fig, message = build_scatter_plot(
                    chart_df,
                    x,
                    y,
                    f"{y} vs {x} in search results",
                    color=color_col,
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(message)
            else:
                st.info("At least two numeric columns are required.")

        elif chart_type == "Correlation heatmap":
            fig, message = build_correlation_heatmap(
                chart_df,
                "Search-result correlation heatmap",
                chart_numeric,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(message)


report_df = df.copy()
summary = get_dataset_summary(report_df)
row_count = summary["row_count"]
column_count = summary["column_count"]
missing_count = summary["missing_count"]
duplicate_count = summary["duplicate_count"]
completeness = summary["completeness"]
numeric_df = report_df[numeric_columns(report_df, include_bool=True)]


with quality_tab:
    st.subheader("Data Quality Report")
    st.caption("These metrics use the original dataset, not the current search results.")

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
        st.success("No missing values in the original dataset.")
    else:
        st.warning("The original dataset contains missing values.")

    st.markdown("#### Duplicate Rows")
    st.metric("Duplicate row rate", f"{summary['duplicate_percent']:.2f}%")


with columns_tab:
    st.subheader("Column Information")
    st.caption("Column counts use the original dataset, not the current search results.")

    column_report = pd.DataFrame({
        "Column": report_df.columns.astype(str),
        "Data Type": [str(dtype) for dtype in report_df.dtypes],
        "Total Rows": [len(report_df)] * len(report_df.columns),
        "Unique Values": report_df.nunique(dropna=True).values,
    })
    st.dataframe(column_report, use_container_width=True, hide_index=True)

    if len(report_df.columns):
        selected_column = st.selectbox(
            "Inspect a column",
            list(report_df.columns),
            key="dl_inspect_column",
        )
        series = get_column_series(report_df, selected_column)
        common = (
            series.value_counts(dropna=False)
            .head(10)
            .rename_axis("Value")
            .reset_index(name="Frequency")
        )
        st.markdown("#### Most Common Values")
        st.dataframe(common, use_container_width=True, hide_index=True)


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
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    else:
        st.info("No numeric columns are available.")

    st.divider()
    st.markdown("#### Categorical Descriptive Statistics")
    categorical_df = report_df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )
    if not categorical_df.empty:
        st.dataframe(categorical_df.describe().T, use_container_width=True)
    else:
        st.info("No categorical columns are available.")


with reports_tab:
    st.subheader("Visual Reports")
    st.caption(
        "These charts use the original dataset. Missing-value details are "
        "available in the Data Quality tab."
    )

    if report_df.empty:
        st.info("No rows are available for visualization.")
    else:
        st.markdown("#### 2. Numeric Distribution")
        report_numeric = numeric_columns(report_df)
        if report_numeric:
            selected_numeric = st.selectbox(
                "Numeric column",
                report_numeric,
                key="dl_report_hist_column",
            )
            fig, message = build_histogram(
                report_df,
                selected_numeric,
                f"Distribution of {selected_numeric}",
                nbins=20,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(message)
        else:
            st.info("No numeric columns are available.")

        st.divider()
        st.markdown("#### 3. Categorical Value Frequencies")
        report_categorical = categorical_columns(report_df)
        if report_categorical:
            selected_category = st.selectbox(
                "Categorical column",
                report_categorical,
                key="dl_report_category_column",
            )
            fig, message = build_category_bar(
                report_df,
                selected_category,
                f"Most common values in {selected_category}",
                limit=20,
                orientation="h",
                category_axis="Value",
                count_axis="Frequency",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(message)
        else:
            st.info("No categorical columns are available.")

        st.divider()
        st.markdown("#### 4. Frequency Distribution")
        frequency_column = st.selectbox(
            "Column for frequency analysis",
            list(report_df.columns),
            key="dl_frequency_column",
        )
        frequency_series = get_column_series(report_df, frequency_column)
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
            fig, message = build_frequency_bar(
                frequency_table,
                f"Frequency distribution of {frequency_column}",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            elif message:
                st.info(message)

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
        st.markdown("#### 5. Numeric Correlation Heatmap")
        fig, message = build_correlation_heatmap_annotated(
            report_df,
            "Correlation between numeric columns",
            report_numeric,
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Correlation ranges from -1 to +1. "
                "Correlation does not necessarily imply causation."
            )
        else:
            st.info(message)


st.divider()
st.caption("DataLens · Data made clearer.")
