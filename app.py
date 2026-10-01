
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# RetailPulse - Demand Forecasting Dashboard
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_QUALITY_REPORT_PATH = PROJECT_DIR / "reports" / "data_quality_report.json"

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "daily_sales_with_promotion_features.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "random_forest_demand_forecast.joblib"
)

FEATURES_PATH = (
    PROJECT_DIR
    / "models"
    / "forecast_features.json"
)

st.set_page_config(
    page_title="RetailPulse | Demand Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RetailPulse")
st.subheader("AI-Powered Customer Analytics & Demand Forecasting")
st.caption("Retail demand monitoring and forecasting")

st.info(
    "Welcome to RetailPulse! This dashboard will display "
    "historical sales trends and machine-learning forecasts."
)

# Check required project files before loading them.
missing_files = [
    path for path in [DATA_PATH, MODEL_PATH, FEATURES_PATH]
    if not path.exists()
]

if missing_files:
    st.error("Some required project files were not found:")
    for path in missing_files:
        st.code(str(path))
    st.stop()

# Load processed daily sales and saved model.
@st.cache_data
def load_sales_data(path):
    data = pd.read_csv(path, parse_dates=["date"])
    return data.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_forecast_model(path):
    return joblib.load(path)


sales_data = load_sales_data(str(DATA_PATH))
forecast_model = load_forecast_model(str(MODEL_PATH))
feature_cols = pd.read_json(FEATURES_PATH, typ="series").tolist()

st.success("Sales data and forecasting model loaded successfully!")

st.write("### Dataset preview")
st.dataframe(sales_data.head(10), width="stretch")

st.write("### Data summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Daily records", f"{len(sales_data):,}")

with col2:
    st.metric(
        "Total units sold",
        f"{sales_data['units_sold'].sum():,.0f}",
    )

with col3:
    st.metric(
        "Forecast model features",
        f"{len(feature_cols)}",
    )
# --------------------------------------------------
# Daily Demand Trend
# --------------------------------------------------

st.divider()
st.header("📈 Daily Demand Trend")

min_date = sales_data["date"].min().date()
max_date = sales_data["date"].max().date()

date_range = st.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered_data = sales_data[
        (sales_data["date"].dt.date >= start_date)
        & (sales_data["date"].dt.date <= end_date)
    ].copy()

    if not filtered_data.empty:
        st.metric(
            "Units sold in selected period",
            f"{filtered_data['units_sold'].sum():,.0f}",
        )

        st.line_chart(
            filtered_data,
            x="date",
            y="units_sold",
            width="stretch",
        )
    else:
        st.warning("No sales records found for this date range.")
# --------------------------------------------------
# AI Demand Forecast
# --------------------------------------------------

st.divider()
st.header("🤖 AI Demand Forecast")
st.write(
    "Generate a daily demand forecast using the saved "
    "RetailPulse Random Forest model."
)

st.warning(
    "Your sales history ends on 2025-12-31. This demonstration "
    "forecast starts on 2026-01-01 and does not represent "
    "a current forecast based on newer sales data. Future "
    "promotion activity is assumed to be zero because no "
    "future promotion schedule is available in this dataset."
)

forecast_days = st.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
)

if st.button("Generate Demand Forecast", type="primary"):

    history = {
        pd.Timestamp(row.date).normalize(): float(row.units_sold)
        for row in sales_data.itertuples(index=False)
    }

    last_date = max(history)
    forecast_rows = []

    for step in range(1, forecast_days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)

        previous_day = forecast_date - pd.Timedelta(days=1)
        previous_week = forecast_date - pd.Timedelta(days=7)
        previous_year_weekday = forecast_date - pd.Timedelta(days=364)

        recent_values = [
            history[forecast_date - pd.Timedelta(days=i)]
            for i in range(1, 8)
        ]

        features = {
            "day_of_week": forecast_date.dayofweek,
            "day_of_month": forecast_date.day,
            "month": forecast_date.month,
            "quarter": forecast_date.quarter,
            "is_weekend": int(forecast_date.dayofweek >= 5),

            # No future promotion schedule is available.
            "active_promotion_count": 0,
            "active_all_target_promotions": 0,
            "active_targeted_promotions": 0,
            "max_scheduled_discount_pct": 0.0,

            "units_lag_1": history[previous_day],
            "units_lag_7": history[previous_week],
            "units_lag_364": history[previous_year_weekday],
            "units_rolling_mean_7": sum(recent_values) / 7,
            "year": forecast_date.year,
        }

        input_row = pd.DataFrame([features])[feature_cols]
        prediction = float(forecast_model.predict(input_row)[0])
        prediction = max(0.0, prediction)

        # Use each forecast as input for subsequent forecast days.
        history[forecast_date] = prediction

        forecast_rows.append({
            "date": forecast_date,
            "forecast_units": round(prediction, 2),
        })

    forecast_df = pd.DataFrame(forecast_rows)

    st.subheader("Forecast Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Forecast period",
            f"{forecast_days} days",
        )

    with col2:
        st.metric(
            "Total forecast units",
            f"{forecast_df['forecast_units'].sum():,.0f}",
        )

    st.line_chart(
        forecast_df,
        x="date",
        y="forecast_units",
        width="stretch",
    )

    st.dataframe(
    forecast_df,
    width="stretch",
    hide_index=True,
)

    csv_data = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        data=csv_data,
        file_name="retailpulse_demand_forecast.csv",
        mime="text/csv",
    )
# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.divider()
st.header("📊 Model Performance")

st.write(
    "Historical evaluation on October–December 2025, "
    "using rolling one-day-ahead predictions."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean Absolute Error", "411.93 units/day")

with col2:
    st.metric("Root Mean Squared Error", "509.59 units/day")

with col3:
    st.metric("Mean Absolute Percentage Error", "2.55%")

st.caption(
    "Evaluation results are from the historical test period. "
    "They are not a guarantee of future accuracy. During testing, "
    "actual previous-day sales were available for lag features; "
    "the multi-day demonstration forecast instead uses its own "
    "previous predictions. Performance may therefore differ."
)

st.subheader("Quarterly Backtesting")

st.write(
    "Across seven historical quarters, the Random Forest model "
    "had a mean quarterly MAPE of 3.26%, compared with 5.24% "
    "for the seasonal-naive baseline. These are averages of "
    "quarter-level MAPE values, not one pooled MAPE."
)
# --------------------------------------------------
# Quarterly Model Comparison
# --------------------------------------------------

st.divider()
st.header("📉 Quarterly Forecast Comparison")

backtest_path = (
    PROJECT_DIR / "reports" / "quarterly_forecast_backtest.csv"
)

if backtest_path.exists():
    import altair as alt

    backtest_df = pd.read_csv(backtest_path)

    chart_data = backtest_df[
        ["quarter", "RF_MAPE_pct", "Baseline_MAPE_pct"]
    ].melt(
        id_vars="quarter",
        var_name="Model",
        value_name="MAPE",
    )

    chart_data["Model"] = chart_data["Model"].map({
        "RF_MAPE_pct": "Random Forest",
        "Baseline_MAPE_pct": "Seasonal Naive",
    })

    st.write(
        "Quarterly MAPE comparison. Lower values indicate "
        "smaller percentage forecast errors."
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("quarter:N", title="Quarter"),
            xOffset=alt.XOffset("Model:N"),
            y=alt.Y("MAPE:Q", title="MAPE (%)", stack=None),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                alt.Tooltip("quarter:N", title="Quarter"),
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("MAPE:Q", title="MAPE (%)", format=".2f"),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(chart, width="stretch")

    st.dataframe(
        backtest_df,
        width="stretch",
        hide_index=True,
    )
else:
    st.warning(
        "Quarterly backtesting file not found: "
        "reports/quarterly_forecast_backtest.csv"
    )

# Customer Segmentation Analysis
st.header("👥 Customer Segmentation Analysis")

segments_path = Path("data/processed/customer_segments.csv")
segment_profiles_path = Path("reports/customer_segment_profiles.csv")

if segments_path.exists() and segment_profiles_path.exists():
    segments_df = pd.read_csv(segments_path)
    segment_profiles = pd.read_csv(segment_profiles_path)

    col1, col2 = st.columns(2)
    col1.metric("Customers Segmented", f"{segments_df['cust_id'].nunique():,}")
    col2.metric("Number of Segments", f"{segments_df['cluster_id'].nunique()}")

    st.subheader("Customer Segment Distribution")

    import altair as alt

    segment_chart = (
        alt.Chart(segment_profiles)
        .mark_bar()
        .encode(
            x=alt.X(
                "customer_count:Q",
                title="Number of Customers",
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y(
                "cluster_id:N",
                title="Customer Segment",
                sort="-x",
            ),
            color=alt.Color(
                "cluster_id:N",
                title="Segment",
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("cluster_id:N", title="Segment"),
                alt.Tooltip("customer_count:Q", title="Customers"),
                alt.Tooltip(
                    "customer_share_pct:Q",
                    title="Customer Share (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=240)
    )

    st.altair_chart(segment_chart, width="stretch")

    st.subheader("Segment Profile Summary")

    display_profiles = segment_profiles.rename(
        columns={
            "cluster_id": "Segment",
            "customer_count": "Customers",
            "avg_recency_days": "Average Recency (Days)",
            "avg_orders": "Average Orders",
            "avg_spend": "Average Spend",
            "median_spend": "Median Spend",
            "avg_units": "Average Units",
            "customer_share_pct": "Customer Share (%)",
        }
    )

    st.dataframe(
        display_profiles.round(2),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Explore Individual Customers")

    available_segments = sorted(segments_df["cluster_id"].dropna().unique())
    selected_segment = st.selectbox(
        "Filter by customer segment",
        options=["All Segments"] + available_segments,
        key="segmentation_filter",
    )

    filtered_segments = segments_df.copy()

    if selected_segment != "All Segments":
        filtered_segments = filtered_segments[
            filtered_segments["cluster_id"] == selected_segment
        ]

    st.caption(f"Showing {len(filtered_segments):,} customers")

    st.dataframe(
        filtered_segments,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        label="Download Customer Segmentation CSV",
        data=filtered_segments.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv",
        key="download_customer_segments",
    )

else:
    st.warning(
        "Customer segmentation files were not found. "
        "Please run the customer segmentation pipeline first."
    )


# --------------------------------------------------
# Customer Retention Analysis
# --------------------------------------------------

st.divider()
st.header("👥 Customer Retention Analysis")

st.write(
    "Explore customer purchase recency, spending, and "
    "rule-based inactivity categories. These categories "
    "are monitoring indicators, not verified churn predictions."
)

retention_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "customer_retention_analysis.csv"
)

retention_summary_path = (
    PROJECT_DIR
    / "reports"
    / "customer_retention_summary.csv"
)

if retention_path.exists() and retention_summary_path.exists():
    retention_df = pd.read_csv(retention_path)
    retention_summary = pd.read_csv(retention_summary_path)

    # Summary metrics
    total_customers = retention_df["customer_id"].nunique()

    extended_count = int(
        retention_df["retention_category"]
        .eq("Extended Inactivity")
        .sum()
    )

    watch_count = int(
        retention_df["retention_category"]
        .eq("Inactivity Watch")
        .sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers Analyzed", f"{total_customers:,}")
    col2.metric("Inactivity Watch", f"{watch_count:,}")
    col3.metric("Extended Inactivity", f"{extended_count:,}")
    st.subheader("Customer Recency Categories")

    import altair as alt

    retention_chart = (
        alt.Chart(retention_summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "customer_count:Q",
                title="Number of Customers",
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y(
                "retention_category:N",
                title="Retention Category",
                sort="-x",
            ),
            tooltip=[
                alt.Tooltip("retention_category:N", title="Category"),
                alt.Tooltip("customer_count:Q", title="Customers"),
                alt.Tooltip(
                    "customer_percentage:Q",
                    title="Customer Share (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(retention_chart, width="stretch")

    st.subheader("Retention Category Summary")

    st.dataframe(
        retention_summary,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Explore Individual Customers")

    available_categories = sorted(
        retention_df["retention_category"].dropna().unique()
    )

    selected_category = st.selectbox(
        "Filter by retention category",
        ["All Categories"] + available_categories,
    )

    filtered_retention = retention_df.copy()

    if selected_category != "All Categories":
        filtered_retention = filtered_retention[
            filtered_retention["retention_category"]
            == selected_category
        ]

    display_columns = [
        "customer_id",
        "recency_days",
        "total_receipts",
        "total_units",
        "total_spend",
        "retention_category",
        "risk_interpretation",
    ]

    st.dataframe(
        filtered_retention[display_columns],
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        label="Download Customer Retention Analysis",
        data=filtered_retention[display_columns].to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_retention_analysis.csv",
        mime="text/csv",
    )

else:
    st.warning(
        "Customer retention files were not found. "
        "Run `python src/customer_retention.py` first."
    )


# Customer Spending Decline Analysis
st.header("💰 Customer Spending Analysis")

spending_path = Path("data/processed/customer_spending_analysis.csv")
spending_summary_path = Path("reports/customer_spending_decline_summary.csv")

if spending_path.exists() and spending_summary_path.exists():
    spending_df = pd.read_csv(spending_path)
    spending_summary = pd.read_csv(spending_summary_path)

    total_customers = spending_df["customer_id"].nunique()
    declining_customers = spending_df[
        spending_df["spending_category"].isin(
            ["Low Decline", "Moderate Decline", "High Decline"]
        )
    ]["customer_id"].nunique()
    high_decline_customers = spending_df[
        spending_df["spending_category"] == "High Decline"
    ]["customer_id"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers Analysed", f"{total_customers:,}")
    col2.metric("Customers with Spending Decline", f"{declining_customers:,}")
    col3.metric("High Spending Decline", f"{high_decline_customers:,}")

    st.caption(
        "Historical comparison: January–June 2025 versus "
        "July–December 2025. These categories describe past spending "
        "changes, not confirmed future churn."
    )

    st.subheader("Spending Change by Customer Category")

    import altair as alt

    spending_chart = (
        alt.Chart(spending_summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "customer_count:Q",
                title="Number of Customers",
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y(
                "spending_category:N",
                title="Spending Category",
                sort="-x",
            ),
            color=alt.Color(
                "spending_category:N",
                title="Category",
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("spending_category:N", title="Category"),
                alt.Tooltip("customer_count:Q", title="Customers"),
                alt.Tooltip(
                    "customer_percentage:Q",
                    title="Customer Share (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "average_spending_change_pct:Q",
                    title="Average Spending Change (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=240)
    )

    st.altair_chart(spending_chart, width="stretch")

    st.subheader("Spending Category Summary")

    display_spending_summary = spending_summary.rename(
        columns={
            "spending_category": "Spending Category",
            "customer_count": "Customers",
            "average_baseline_spend": "Average Baseline Spend",
            "average_followup_spend": "Average Follow-up Spend",
            "average_spending_change_pct": "Average Change (%)",
            "customer_percentage": "Customer Share (%)",
        }
    )

    st.dataframe(
        display_spending_summary.round(2),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Explore Customer Spending")

    spending_categories = sorted(
        spending_df["spending_category"].dropna().unique().tolist()
    )

    selected_spending_category = st.selectbox(
        "Filter by spending category",
        options=["All Categories"] + spending_categories,
        key="spending_analysis_filter",
    )

    filtered_spending = spending_df.copy()

    if selected_spending_category != "All Categories":
        filtered_spending = filtered_spending[
            filtered_spending["spending_category"]
            == selected_spending_category
        ]

    st.caption(f"Showing {len(filtered_spending):,} customers")

    st.dataframe(
        filtered_spending,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        label="Download Customer Spending Analysis CSV",
        data=filtered_spending.to_csv(index=False).encode("utf-8"),
        file_name="customer_spending_analysis.csv",
        mime="text/csv",
        key="customer_spending_analysis_download",
    )
# Inventory Optimization
st.header("📦 Inventory Optimization")

inventory_path = Path("data/processed/inventory_recommendations.csv")
inventory_summary_path = Path("reports/inventory_optimization_summary.csv")

if inventory_path.exists() and inventory_summary_path.exists():
    inventory_df = pd.read_csv(inventory_path)
    inventory_summary = pd.read_csv(inventory_summary_path)

    total_pairs = len(inventory_df)
    urgent_stockouts = int(
        (inventory_df["recommendation_status"] == "STOCKOUT_URGENT").sum()
    )
    pairs_to_replenish = int(
        (inventory_df["recommended_order_qty"] > 0).sum()
    )
    total_order_units = int(inventory_df["recommended_order_qty"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Store–SKU Pairs", f"{total_pairs:,}")
    col2.metric("Urgent Stockouts", f"{urgent_stockouts:,}")
    col3.metric("Pairs to Replenish", f"{pairs_to_replenish:,}")
    col4.metric("Suggested Order Units", f"{total_order_units:,}")

    demand_window = inventory_summary["demand_window"].iloc[0]
    target_cover = int(
        inventory_summary["target_cover_days_assumption"].iloc[0]
    )

    st.caption(
        f"Historical demand window: {demand_window}. "
        f"Target stock cover assumption: {target_cover} days. "
        "These are planning suggestions based on historical data, "
        "not live stock levels or confirmed purchase orders. "
        "Verify supplier lead times and incoming inventory before ordering."
    )

    st.subheader("Inventory Status Overview")
    status_chart = inventory_summary.set_index(
        "recommendation_status"
    )[["inventory_pairs"]]
    st.bar_chart(status_chart)

    st.subheader("Explore Inventory Recommendations")

    status_options = sorted(
        inventory_df["recommendation_status"].dropna().unique().tolist()
    )
    city_options = sorted(inventory_df["city"].dropna().unique().tolist())

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_status = st.selectbox(
            "Filter by recommendation status",
            ["All Statuses"] + status_options,
            key="inventory_status_filter",
        )

    with filter_col2:
        selected_city = st.selectbox(
            "Filter by city",
            ["All Cities"] + city_options,
            key="inventory_city_filter",
        )

    stores_for_city = inventory_df
    if selected_city != "All Cities":
        stores_for_city = stores_for_city[
            stores_for_city["city"] == selected_city
        ]
    store_options = sorted(
        stores_for_city["store_name"].dropna().unique().tolist()
    )

    with filter_col3:
        selected_store = st.selectbox(
            "Filter by store",
            ["All Stores"] + store_options,
            key="inventory_store_filter",
        )

    filtered_inventory = inventory_df.copy()

    if selected_status != "All Statuses":
        filtered_inventory = filtered_inventory[
            filtered_inventory["recommendation_status"] == selected_status
        ]

    if selected_city != "All Cities":
        filtered_inventory = filtered_inventory[
            filtered_inventory["city"] == selected_city
        ]

    if selected_store != "All Stores":
        filtered_inventory = filtered_inventory[
            filtered_inventory["store_name"] == selected_store
        ]

    display_columns = [
        "store_name",
        "city",
        "sku_id",
        "sku_name",
        "category",
        "stock_on_hand",
        "reorder_point",
        "safety_stock",
        "units_90d",
        "stock_cover_days",
        "recommended_order_qty",
        "recommendation_status",
        "recommendation_note",
    ]

    st.caption(
        f"Showing {len(filtered_inventory):,} of {total_pairs:,} "
        "store–SKU pairs"
    )

    st.dataframe(
        filtered_inventory[display_columns],
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        label="Download Inventory Recommendations CSV",
        data=filtered_inventory.to_csv(index=False).encode("utf-8"),
        file_name="inventory_recommendations.csv",
        mime="text/csv",
        key="inventory_optimization_download",
    )

else:
    st.warning(
        "Inventory optimization files were not found. "
        "Run `python -m src.inventory_optimization` first."
    )
st.divider()
st.header("🔎 Data Quality Checks")

if DATA_QUALITY_REPORT_PATH.exists():
    try:
        with open(DATA_QUALITY_REPORT_PATH, "r", encoding="utf-8") as report_file:
            quality_report = json.load(report_file)

            quality_result = quality_report.get("quality_result", "UNKNOWN")

        if quality_result == "REVIEW_REQUIRED":
            st.warning(
                "Review required: the report detected data-quality "
                "findings that should be investigated."
            )
        elif quality_result == "NO_ISSUES_DETECTED":
            st.success("No data-quality issues were detected.")
        else:
            st.error(
                f"Unexpected data-quality report status: {quality_result}"
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sales Records Scanned",
            f"{quality_report.get('total_rows', 0):,}"
        )
        col2.metric(
            "Excess Duplicate Rows",
            f"{quality_report.get('excess_exact_duplicate_rows', 0):,}"
        )
        col3.metric(
            "Sales Value Mismatches",
            f"{quality_report.get('sales_value_mismatches', 0):,}"
        )

        with st.expander("View Detailed Quality Findings"):
            st.write("**Missing required columns**")
            st.write(quality_report.get("missing_required_columns", []))

            st.write("**Missing values by column**")
            st.json(quality_report.get("missing_value_counts", {}))

            st.write("**Invalid values**")
            st.json(quality_report.get("invalid_value_counts", {}))

            st.write(
                "**Overall result:**",
                quality_report.get("quality_result", "UNKNOWN")
            )

    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read the data quality report: {error}")
else:
    st.info(
        "No saved data quality report found. Run "
        "`python -m src.data_quality_checks` in the terminal first."
    )
