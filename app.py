
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# RetailPulse - Demand Forecasting Dashboard
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

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
    backtest_df = pd.read_csv(backtest_path)

    st.write(
        "Compare quarterly MAPE (lower values indicate "
        "smaller percentage forecast errors)."
    )

    chart_data = backtest_df.set_index("quarter")[
        ["RF_MAPE_pct", "Baseline_MAPE_pct"]
    ].rename(
        columns={
            "RF_MAPE_pct": "Random Forest",
            "Baseline_MAPE_pct": "Seasonal Naive",
        }
    )

    st.bar_chart(chart_data, width="stretch")

    st.dataframe(
        backtest_df,
        width="stretch",
        hide_index=True,
    )
else:
    st.warning(
        "Quarterly backtesting results were not found. "
        "Check reports/quarterly_forecast_backtest.csv."
    )
    chart_data = backtest_df[
        ["quarter", "RF_MAPE_pct", "Baseline_MAPE_pct"]
    ].melt(
        id_vars="quarter",
        value_vars=["RF_MAPE_pct", "Baseline_MAPE_pct"],
        var_name="Model",
        value_name="MAPE (%)",
    )

    chart_data["Model"] = chart_data["Model"].map({
        "RF_MAPE_pct": "Random Forest",
        "Baseline_MAPE_pct": "Seasonal Naive",
    })

    import altair as alt

    comparison_chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("quarter:N", title="Quarter"),
            xOffset=alt.XOffset("Model:N"),
            y=alt.Y("MAPE (%):Q", title="MAPE (%)", stack=None),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["quarter", "Model", "MAPE (%)"],
        )
        .properties(height=400)
    )

    st.altair_chart(comparison_chart, width="stretch")
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