import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="PJM Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD FORECAST DATA
# ---------------------------------------------------------

@st.cache_data
def load_forecast_data():

    hourly_forecast = pd.read_csv(
        "pjm_30_day_hourly_forecast.csv"
    )

    daily_forecast = pd.read_csv(
        "pjm_30_day_daily_forecast.csv"
    )

    model_results = pd.read_csv(
        "model_comparison_results.csv"
    )

    forecast_history = pd.read_csv(
        "forecast_history_168.csv"
    )

    # Convert date columns

    hourly_forecast["Datetime"] = pd.to_datetime(
        hourly_forecast["Datetime"]
    )

    daily_forecast["Date"] = pd.to_datetime(
        daily_forecast["Date"]
    )

    forecast_history["Datetime"] = pd.to_datetime(
        forecast_history["Datetime"]
    )

    return (
        hourly_forecast,
        daily_forecast,
        model_results,
        forecast_history
    )


(
    generated_forecast,
    generated_daily_forecast,
    model_results,
    forecast_history
) = load_forecast_data()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("⚡ PJM Energy Forecast")

st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Model Performance",
        "30-Day Forecast",
        "Forecast Data"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.title("⚡ PJM Hourly Energy Consumption Forecast")

    st.write(
        "Machine Learning based forecasting of PJM "
        "electricity consumption for the next 30 days."
    )

    st.header("Project Overview")

    st.write(
        """
        The objective of this project is to analyze
        historical PJM electricity consumption data
        and forecast future energy consumption.

        The project uses time-based features, lag features,
        rolling averages, and machine learning models.

        Random Forest Regression was selected as the
        final model after comparing multiple models.
        """
    )

    # -----------------------------------------------------
    # FORECAST SUMMARY
    # -----------------------------------------------------

    st.header("Forecast Summary")

    total_hours = len(generated_forecast)

    total_days = len(generated_daily_forecast)

    average_consumption = (
        generated_forecast["Predicted_MW"].mean()
    )

    maximum_consumption = (
        generated_forecast["Predicted_MW"].max()
    )

    minimum_consumption = (
        generated_forecast["Predicted_MW"].min()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Forecast Hours",
            total_hours
        )

    with col2:
        st.metric(
            "Forecast Days",
            total_days
        )

    with col3:
        st.metric(
            "Average MW",
            f"{average_consumption:.2f}"
        )

    with col4:
        st.metric(
            "Maximum MW",
            f"{maximum_consumption:.2f}"
        )

    with col5:
        st.metric(
            "Minimum MW",
            f"{minimum_consumption:.2f}"
        )

    # -----------------------------------------------------
    # DAILY FORECAST
    # -----------------------------------------------------

    st.header("Daily Forecast")

    st.line_chart(
        generated_daily_forecast.set_index(
            "Date"
        )["Average_Predicted_MW"]
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Comparison of the models evaluated during "
        "the model-building phase."
    )

    # -----------------------------------------------------
    # MODEL RESULTS TABLE
    # -----------------------------------------------------

    st.dataframe(
        model_results,
        width="stretch"
    )

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    st.subheader("Model Comparison")

    st.bar_chart(
        model_results.set_index(
            "Model"
        )[["MAE", "RMSE"]]
    )

    # -----------------------------------------------------
    # FINAL MODEL
    # -----------------------------------------------------

    st.subheader("Final Model")

    st.success(
        "Random Forest Regression was selected "
        "as the final model."
    )

    rf_result = model_results[
        model_results["Model"] == "Random Forest"
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{rf_result['MAE']:.2f} MW"
        )

    with col2:

        st.metric(
            "RMSE",
            f"{rf_result['RMSE']:.2f} MW"
        )

    with col3:

        st.metric(
            "MAPE",
            f"{rf_result['MAPE (%)']:.2f}%"
        )

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.subheader("Model Information")

    st.write(
        """
        The Random Forest model was selected because it
        achieved the lowest error values among the models
        tested during the project.

        Final Model Performance:

        • MAE: 59.36 MW
        • RMSE: 78.35 MW
        • MAPE: 1.04%
        """
    )


# =========================================================
# 30-DAY FORECAST
# =========================================================

elif page == "30-Day Forecast":

    st.title("📈 30-Day Energy Consumption Forecast")

    # -----------------------------------------------------
    # HOURLY FORECAST
    # -----------------------------------------------------

    st.subheader("Hourly Forecast")

    st.line_chart(
        generated_forecast.set_index(
            "Datetime"
        )["Predicted_MW"]
    )

    # -----------------------------------------------------
    # DAILY FORECAST
    # -----------------------------------------------------

    st.subheader("Daily Average Forecast")

    st.line_chart(
        generated_daily_forecast.set_index(
            "Date"
        )["Average_Predicted_MW"]
    )


# =========================================================
# FORECAST DATA
# =========================================================

elif page == "Forecast Data":

    st.title("📋 Forecast Data")

    st.write(
        "Hourly energy consumption predictions "
        "for the next 30 days."
    )

    # -----------------------------------------------------
    # DATA TABLE
    # -----------------------------------------------------

    st.dataframe(
        generated_forecast,
        width="stretch"
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.subheader("Download Forecast")

    csv_data = generated_forecast.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download 30-Day Forecast",
        data=csv_data,
        file_name="pjm_30_day_hourly_forecast.csv",
        mime="text/csv",
        width="stretch"
    )

    st.success(
        "30-day forecast is ready for download."
    )