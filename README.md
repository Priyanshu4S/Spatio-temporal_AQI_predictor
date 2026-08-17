# CO Forecasting using OLS and ARMA

## Overview

This project analyzes and forecasts carbon monoxide (CO) concentration using time-series analysis, Ordinary Least Squares (OLS) regression, and ARMA/ARIMA residual modeling.

The workflow includes exploratory time-series analysis, construction and evaluation of a design-matrix-based OLS model, ARMA modeling of OLS residuals, and out-of-sample forecasting.

## Project Structure

```text
CO-Forecasting-OLS-ARMA/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── time_series_analysis.py
│   ├── ols_arma_model.py
│   └── ols_arma_one_step.py
│
└── data/
    ├── data_station1.csv
    ├── 2019-2022_co_ds1.csv
    ├── Design_matrix_35040x390.csv
    ├── data_station1_2023.csv
    └── test_matrix.csv
```

## Dataset

The project uses hourly air-quality data containing pollutant measurements including:

- CO
- PM2.5
- PM10
- NOx
- SO2
- O3

The main station dataset covers the period from January 2010 to March 2023.

For the forecasting model, the data is divided into training and test datasets:

- **Training:** 2019–2022 CO data
- **Testing:** 2023 CO data

The project also uses pre-generated design matrices for the OLS model.

## Methodology

### 1. Time-Series Analysis

`time_series_analysis.py` performs exploratory analysis of the CO time series, including:

- Time-series visualization
- Day, hour, and month feature extraction
- Annual and daily trigonometric features
- Augmented Dickey-Fuller (ADF) test
- Seasonal decomposition
- ACF and PACF analysis
- Kruskal-Wallis test for monthly seasonality
- Monthly boxplot analysis

### 2. OLS Regression

The forecasting model uses a design matrix containing day and hour variables.

The model:

1. Loads the training CO data and design matrix.
2. Removes `Day_365` and `Hour_24` to avoid the dummy-variable trap.
3. Checks the rank of the design matrix.
4. Estimates the OLS coefficients using the normal equations.
5. Generates fitted values and residuals.
6. Evaluates the model using RMSE, MAE, and R².

### 3. ARMA Residual Modeling

The residuals from the OLS model are modeled using an ARIMA model with order `(1,0,1)`.

The fitted ARMA component is then combined with the OLS prediction:

```text
Final prediction = OLS prediction + ARMA residual prediction
```

This allows the regression component to capture systematic temporal patterns while the ARMA component models the remaining time-dependent structure in the residuals.

### 4. Out-of-Sample Forecasting

The project evaluates the model on a separate 2023 test dataset.

The main forecasting script uses:

- OLS predictions on the test design matrix
- ARMA residual forecasting
- Combined OLS + ARMA predictions

The one-step-ahead version additionally evaluates online updating using the observed test residuals.

## Evaluation Metrics

The models are evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**

The scripts print the evaluation metrics for the training and test datasets.

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the exploratory time-series analysis:

```bash
python src/time_series_analysis.py
```

Run the main OLS + ARMA forecasting model:

```bash
python src/ols_arma_model.py
```

Run the one-step-ahead OLS + ARMA model:

```bash
python src/ols_arma_one_step.py
```

## Notes

The Python scripts should use paths relative to the repository's `data/` directory rather than machine-specific paths. This allows the project to be reproduced on another computer after cloning the repository.

The design matrices are provided as CSV files and are used directly by the OLS forecasting scripts.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- Scikit-learn
- SciPy
- Seaborn
