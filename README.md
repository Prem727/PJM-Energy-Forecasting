# ⚡ PJM Hourly Energy Consumption Forecast

## 📌 Project Overview

This project focuses on forecasting hourly electricity consumption using historical PJM energy consumption data.

The project uses time-based features, lag features, rolling averages, and machine learning models to predict future energy consumption.

The final application provides a 30-day hourly energy consumption forecast using a trained Random Forest Regression model.

---

## 🎯 Business Objective

The main objectives of this project are:

* Analyze historical PJM electricity consumption.
* Understand hourly, daily, weekly, monthly, and seasonal consumption patterns.
* Identify the effect of weekends and holidays on energy consumption.
* Build machine learning models for energy consumption forecasting.
* Evaluate different models using MAE, RMSE, and MAPE.
* Forecast energy consumption for the next 30 days.
* Deploy the forecasting model using Streamlit.

---

## 📊 Dataset

The dataset contains hourly PJM electricity consumption data.

### Dataset Columns

| Column   | Description                               |
| -------- | ----------------------------------------- |
| Datetime | Date and time of energy consumption       |
| PJMW_MW  | Electricity consumption in megawatts (MW) |

The original dataset contains approximately 143,206 observations.

The data covers the period from 2002 to 2018.

---

## 🔎 Exploratory Data Analysis

The following areas were analyzed:

* Dataset structure
* Data types
* Missing values
* Duplicate records
* Datetime patterns
* Overall consumption trends
* Hourly consumption patterns
* Daily consumption patterns
* Weekly consumption patterns
* Weekday vs weekend consumption
* Monthly consumption
* Yearly consumption
* Seasonal patterns
* Summer vs winter patterns
* Holiday consumption
* Outlier analysis

---

## 🛠️ Feature Engineering

The following features were created for machine learning.

### Time Features

* Hour
* DayOfWeek
* Month
* Year
* IsWeekend
* IsHoliday

### Lag Features

* Lag_1
* Lag_24
* Lag_168

### Rolling Features

* RollingMean_24
* RollingMean_168

These features help the model understand recent consumption patterns and seasonal behavior.

---

## 🤖 Machine Learning Models

The following models were evaluated:

1. Baseline Model
2. Linear Regression
3. Random Forest Regression
4. Gradient Boosting Regression

The data was divided chronologically, with the last year used as the test period.

A random train-test split was not used because this is a time-series forecasting problem.

---

## 📈 Model Evaluation

The models were evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* MAPE — Mean Absolute Percentage Error

### Final Model

Random Forest Regression was selected as the final model based on the model comparison performed during the project.

### Random Forest Results

| Metric |   Result |
| ------ | -------: |
| MAE    | 59.36 MW |
| RMSE   | 78.35 MW |
| MAPE   |    1.04% |

---

## 🔮 30-Day Forecast

The final application generates:

* 720 hourly predictions
* 30 daily average predictions

The forecasting application uses the trained Random Forest model and the latest historical energy consumption values.

---

## 💻 Streamlit Application

The application contains four main sections.

### 1. Overview

Displays:

* Project information
* Forecast summary
* Average predicted consumption
* Maximum predicted consumption
* Minimum predicted consumption
* Daily forecast chart

### 2. Model Performance

Displays:

* Model comparison
* MAE
* RMSE
* MAPE
* Final Random Forest model status

### 3. 30-Day Forecast

Displays:

* Hourly forecast
* Daily average forecast

### 4. Forecast Data

Displays:

* Complete hourly forecast table
* Download option for the 30-day forecast

---

# ▶️ How to Run the Project

Follow these steps to run the project on your computer.

## Step 1: Install Python

Make sure Python is installed on your computer.

Check the Python version using:

```bash
python --version
```

---

## Step 2: Open the Project Folder

Open the project folder in VS Code:

```text
C:\Energy_Forecasting
```

The folder should contain the required project files.

---

## Step 3: Open the Terminal

In VS Code:

**Terminal → New Terminal**

The terminal should open inside the project folder.

You can verify the location using:

```bash
cd
```

---

## Step 4: Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

This installs the libraries required by the project.

The main libraries are:

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## Step 5: Run the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

---

## Step 6: Open the Application

After running the command, Streamlit will start the application.

The terminal will show a local address similar to:

```text
http://localhost:8501
```

Open this address in your web browser.

---

## Step 7: Use the Application

Use the sidebar to navigate between:

* Overview
* Model Performance
* 30-Day Forecast
* Forecast Data

You can view the forecast charts, model performance, forecast table, and download the hourly forecast as a CSV file.

---

## 🛑 How to Stop the Application

To stop the Streamlit application, go to the terminal and press:

```text
Ctrl + C
```

---

## 📁 Project Structure

```text
Energy_Forecasting/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── forecast_features.json
├── forecast_history_168.csv
├── model_comparison_results.csv
│
├── pjm_30_day_daily_forecast.csv
├── pjm_30_day_hourly_forecast.csv
│
└── pjm_energy_forecasting_model.pkl
```

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Google Colab
* Git
* GitHub

---

## 📌 Project Outcome

The project analyzes historical hourly energy consumption and uses machine learning to forecast future electricity demand.

The final Streamlit application provides an interactive way to view model performance, 30-day forecasts, and downloadable forecast data.
