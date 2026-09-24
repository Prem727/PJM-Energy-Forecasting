import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PJM Energy Forecast",
    page_icon="⚡",
    layout="wide"
)


# ==========================================================
# LOAD TRAINED RANDOM FOREST MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model_url = "https://huggingface.co/prem727/pjm-energy-forecasting-model/resolve/main/pjm_energy_forecasting_model.pkl"

    model_path = "pjm_energy_forecasting_model.pkl"

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            model_url,
            model_path
        )

    return joblib.load(model_path)


# Load the trained model
final_model = load_model()


# ==========================================================
# LOAD EXISTING FORECAST DATA
# ==========================================================

hourly_forecast = pd.read_csv(
    "pjm_30_day_hourly_forecast.csv"
)

daily_forecast = pd.read_csv(
    "pjm_30_day_daily_forecast.csv"
)

model_results = pd.read_csv(
    "model_comparison_results.csv"
)


# ==========================================================
# LOAD LATEST 168 HISTORICAL VALUES
# ==========================================================

forecast_history = pd.read_csv(
    "forecast_history_168.csv"
)

forecast_history["Datetime"] = pd.to_datetime(
    forecast_history["Datetime"]
)


# ==========================================================
# CONVERT DATE COLUMNS
# ==========================================================

hourly_forecast["Datetime"] = pd.to_datetime(
    hourly_forecast["Datetime"]
)

daily_forecast["Date"] = pd.to_datetime(
    daily_forecast["Date"]
)


# ==========================================================
# GENERATE FUTURE FORECAST USING TRAINED MODEL
# ==========================================================

def generate_forecast(model, history_df, hours=720):

    history = history_df.copy()

    history = history.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    values = history["PJMW_MW"].tolist()

    last_datetime = history["Datetime"].iloc[-1]

    future_predictions = []

    future_dates = []


    # ======================================================
    # FORECAST EACH FUTURE HOUR
    # ======================================================

    for i in range(hours):

        future_datetime = (
            last_datetime
            + pd.Timedelta(hours=i + 1)
        )


        # ==================================================
        # CREATE TIME FEATURES
        # ==================================================

        hour = future_datetime.hour

        day_of_week = future_datetime.dayofweek

        month = future_datetime.month

        year = future_datetime.year

        is_weekend = int(
            day_of_week >= 5
        )


        # ==================================================
        # CREATE HOLIDAY FEATURE
        # ==================================================

        is_holiday = 0


        # ==================================================
        # CREATE LAG FEATURES
        # ==================================================

        lag_1 = values[-1]

        lag_24 = values[-24]

        lag_168 = values[-168]


        # ==================================================
        # CREATE ROLLING MEAN FEATURES
        # ==================================================

        rolling_mean_24 = np.mean(
            values[-24:]
        )

        rolling_mean_168 = np.mean(
            values[-168:]
        )


        # ==================================================
        # CREATE MODEL INPUT
        # ==================================================

        features = pd.DataFrame([{

            "Hour": hour,

            "DayOfWeek": day_of_week,

            "Month": month,

            "Year": year,

            "IsWeekend": is_weekend,

            "IsHoliday": is_holiday,

            "Lag_1": lag_1,

            "Lag_24": lag_24,

            "Lag_168": lag_168,

            "RollingMean_24": rolling_mean_24,

            "RollingMean_168": rolling_mean_168

        }])


        # ==================================================
        # GENERATE PREDICTION
        # ==================================================

        prediction = model.predict(
            features
        )[0]


        # ==================================================
        # STORE PREDICTION
        # ==================================================

        future_predictions.append(
            prediction
        )

        future_dates.append(
            future_datetime
        )


        # ==================================================
        # ADD PREDICTION TO HISTORY
        # ==================================================

        values.append(
            prediction
        )


    # ======================================================
    # CREATE FORECAST DATAFRAME
    # ======================================================

    forecast_df = pd.DataFrame({

        "Datetime": future_dates,

        "Predicted_MW": future_predictions

    })

    return forecast_df


# ==========================================================
# GENERATE 30-DAY HOURLY FORECAST
# ==========================================================

generated_forecast = generate_forecast(
    final_model,
    forecast_history,
    hours=720
)


# ==========================================================
# CREATE DAILY FORECAST
# ==========================================================

generated_forecast["Date"] = (
    generated_forecast["Datetime"].dt.date
)


generated_daily_forecast = (
    generated_forecast
    .groupby("Date")["Predicted_MW"]
    .mean()
    .reset_index()
)


generated_daily_forecast.rename(
    columns={
        "Predicted_MW": "Average_Predicted_MW"
    },
    inplace=True
)


generated_daily_forecast["Date"] = pd.to_datetime(
    generated_daily_forecast["Date"]
)


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title(
    "⚡ PJM Energy Forecast"
)

st.sidebar.write(
    "Navigation"
)


page = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Model Performance",
        "30-Day Forecast",
        "Forecast Data"
    ]
)


# ==========================================================
# OVERVIEW PAGE
# ==========================================================

if page == "Overview":

    st.title(
        "⚡ PJM Hourly Energy Consumption Forecast"
    )

    st.write(
        "Machine Learning based forecasting of PJM "
        "electricity consumption for the next 30 days."
    )


    # ======================================================
    # PROJECT OVERVIEW
    # ======================================================

    st.header(
        "Project Overview"
    )

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


    # ======================================================
    # FORECAST SUMMARY
    # ======================================================

    st.header(
        "Forecast Summary"
    )

    total_hours = len(
        generated_forecast
    )

    total_days = len(
        generated_daily_forecast
    )

    average_consumption = generated_forecast[
        "Predicted_MW"
    ].mean()

    maximum_consumption = generated_forecast[
        "Predicted_MW"
    ].max()

    minimum_consumption = generated_forecast[
        "Predicted_MW"
    ].min()


    # ======================================================
    # DISPLAY FORECAST METRICS
    # ======================================================

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


    # ======================================================
    # DAILY FORECAST CHART
    # ======================================================

    st.header(
        "Daily Forecast"
    )

    st.line_chart(
        generated_daily_forecast.set_index(
            "Date"
        )["Average_Predicted_MW"]
    )


# ==========================================================
# MODEL PERFORMANCE PAGE
# ==========================================================

elif page == "Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Comparison of the models evaluated during "
        "the model-building phase."
    )


    # ======================================================
    # MODEL RESULTS TABLE
    # ======================================================

    st.dataframe(
        model_results,
        use_container_width=True
    )


    # ======================================================
    # MODEL COMPARISON CHART
    # ======================================================

    st.subheader(
        "Model Comparison"
    )

    st.bar_chart(
        model_results.set_index("Model")[
            ["MAE", "RMSE"]
        ]
    )


    # ======================================================
    # FINAL MODEL
    # ======================================================

    st.subheader(
        "Final Model"
    )

    st.success(
        "Random Forest Regression was selected as the "
        "final model."
    )

    st.write(
        "Random Forest achieved the following results:"
    )


    # ======================================================
    # GET RANDOM FOREST RESULTS
    # ======================================================

    rf_result = model_results[
        model_results["Model"] == "Random Forest"
    ].iloc[0]


    # ======================================================
    # DISPLAY MODEL METRICS
    # ======================================================

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


    # ======================================================
    # TRAINED MODEL STATUS
    # ======================================================

    st.subheader(
        "Trained Model Status"
    )

    try:

        test_features = pd.DataFrame([{

            "Hour": 12,

            "DayOfWeek": 0,

            "Month": 1,

            "Year": 2018,

            "IsWeekend": 0,

            "IsHoliday": 0,

            "Lag_1": 5500,

            "Lag_24": 5400,

            "Lag_168": 5600,

            "RollingMean_24": 5500,

            "RollingMean_168": 5600

        }])


        # ==================================================
        # TEST MODEL PREDICTION
        # ==================================================

        test_prediction = final_model.predict(
            test_features
        )[0]


        st.success(
            "Random Forest model loaded successfully."
        )

        st.write(
            f"Test prediction generated by the trained model: "
            f"{test_prediction:.2f} MW"
        )


    except Exception as e:

        st.error(
            f"Model loading or prediction failed: {e}"
        )


# ==========================================================
# 30-DAY FORECAST PAGE
# ==========================================================

elif page == "30-Day Forecast":

    st.title(
        "📈 30-Day Energy Consumption Forecast"
    )


    # ======================================================
    # HOURLY FORECAST
    # ======================================================

    st.subheader(
        "Hourly Forecast"
    )

    st.line_chart(
        generated_forecast.set_index(
            "Datetime"
        )["Predicted_MW"]
    )


    # ======================================================
    # DAILY AVERAGE FORECAST
    # ======================================================

    st.subheader(
        "Daily Average Forecast"
    )

    st.line_chart(
        generated_daily_forecast.set_index(
            "Date"
        )["Average_Predicted_MW"]
    )


# ==========================================================
# FORECAST DATA PAGE
# ==========================================================

elif page == "Forecast Data":

    st.title(
        "📋 Forecast Data"
    )

    st.write(
        "Hourly energy consumption predictions "
        "for the next 30 days."
    )


    # ======================================================
    # FORECAST DATA TABLE
    # ======================================================

    st.dataframe(
        generated_forecast,
        use_container_width=True
    )


    # ======================================================
    # DOWNLOAD FORECAST
    # ==========================================================

    st.subheader(
        "Download Forecast"
    )

    csv_data = generated_forecast.to_csv(
        index=False
    )


    # ======================================================
    # DOWNLOAD BUTTON
    # ======================================================

    st.download_button(
        label="📥 Download 30-Day Forecast",
        data=csv_data,
        file_name="pjm_30_day_hourly_forecast.csv",
        mime="text/csv"
    )

    st.success(
        "30-day forecast is ready for download."
    )