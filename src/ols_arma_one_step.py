# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:46:57 2026

@author: priya
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from pathlib import Path

# This file is stored in src/, so the data folder is one level above it.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

data = pd.read_csv(DATA_DIR / "2019-2022_co_ds1.csv")
matrix = pd.read_csv(DATA_DIR / "Design_matrix_35040x390.csv")

y = data['co'].values
X = matrix.copy()


# Add intercept
#X.insert(0, "intercept", 1)

# Remove any zero-variance columns (e.g., leap day if unused)
#zero_var_cols = [c for c in X.columns if X[c].std() == 0]
#X = X.drop(columns=zero_var_cols)
X = X.drop(columns=['Day_365'])

# Drop one hour dummy
X = X.drop(columns=['Hour_24'])

X_vals = X.values

print("Shape:", X_vals.shape)
print("Rank :", np.linalg.matrix_rank(X_vals))
# ── Step 5: Verify full rank ──
X_vals = X.values
rank = np.linalg.matrix_rank(X_vals)
print(f"\nFinal shape: {X_vals.shape}")
print(f"Rank: {rank}  ← must equal number of columns ({X_vals.shape[1]}) to be invertible")

# ── Step 6: OLS ──
if rank == X_vals.shape[1]:
    XtX = X_vals.T @ X_vals
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X_vals.T @ y)
    y_hat = X_vals @ beta
    epsilon = y - y_hat

    print("\nBeta shape:", beta.shape)
    print("Beta (first 5):", beta[:5])
    print("Residual std (ε):", np.std(epsilon).round(4))
    print("R² score:", round(1 - np.var(epsilon) / np.var(y), 4))
else:
        print("❌ Still not full rank — use pinv as fallback:")
        beta = np.linalg.pinv(X_vals.T @ X_vals) @ (X_vals.T @ y)

# --- Optional cross-verify using statsmodels ---
model = sm.OLS(y, X).fit()
print("\nStatsmodels summary:")
print(model.summary())

from statsmodels.tsa.arima.model import ARIMA

arma_model = ARIMA(epsilon, order=(1,0,1))
arma_fit = arma_model.fit()
data['arma_fitted'] = arma_fit.fittedvalues
data['epsilon'] = epsilon
residual_forecast = arma_fit.forecast(steps=168)
data['y_hat'] = y_hat
data ['final_forecast'] = y_hat + arma_fit.fittedvalues

#######################complete model r2##################
# ----- Combined Regression + ARMA fitted values -----
combined_fitted = y_hat + arma_fit.fittedvalues

# Residuals of the combined model
combined_residuals = y - combined_fitted

# R-squared of the combined model
r2_combined = 1 - np.var(combined_residuals) / np.var(y)

print("\nCombined Model R²:", round(r2_combined, 4))

# (Optional) Save to dataframe
data['combined_fitted'] = combined_fitted

import matplotlib.pyplot as plt

'''plt.figure(figsize=(12,5))

plt.plot(data.datetime, data['epsilon'], label="Actual residual")
plt.plot(data.datetime, data['arma_fitted'], label="ARMA fitted")

plt.legend()
plt.title("Residual vs ARMA Model Fit")
plt.xlabel("Time")
plt.ylabel("Residual")

plt.show()

plt.figure(figsize=(12,5))

plt.plot(data.datetime, data['co'], label="Actual data")
plt.plot(data.datetime, data['y_hat'], label="y_hat")

plt.legend()
plt.title("actual vs base model")
plt.xlabel("Time")
plt.ylabel("conc")

plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

series = data['epsilon']

fig, ax = plt.subplots(1,2, figsize=(12,4))

plot_acf(series, lags=80, ax=ax[0])
plot_pacf(series, lags=80, ax=ax[1])

plt.show()


plt.figure(figsize=(12,5))

plt.plot(data.datetime, y, label='Actual')
plt.plot(data.datetime, combined_fitted, label='Regression + ARMA')

plt.legend()
plt.title("Actual vs Combined Model")
plt.xlabel("Time")
plt.ylabel("CO Concentration")
plt.show()

#########################beta plos#######################
# Convert beta to a Series with column names as index
beta_series = pd.Series(beta, index=X.columns)

# Extract coefficients
day_beta = beta_series[beta_series.index.str.startswith('Day_')]
hour_beta = beta_series[beta_series.index.str.startswith('Hour_')]

plt.figure(figsize=(14,5))

day_numbers = [int(col.split('_')[1]) for col in day_beta.index]

plt.plot(day_numbers, day_beta.values, marker='o', markersize=2)

plt.xlabel("Day of Year")
plt.ylabel("Beta Value")
plt.title("Regression Coefficients for Day Variables")
plt.grid(True)

plt.show()

plt.figure(figsize=(10,5))

hour_numbers = [int(col.split('_')[1]) for col in hour_beta.index]

plt.plot(hour_numbers, hour_beta.values, marker='o')

plt.xlabel("Hour of Day")
plt.ylabel("Beta Value")
plt.title("Regression Coefficients for Hour Variables")
plt.xticks(range(24))
plt.grid(True)

plt.show()'''

from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Step 1: Load the Test Data ──
# Assuming abc.csv contains your test targets and you have a corresponding test design matrix
# ── Step 1: Load the Test Data ──
test_data = pd.read_csv(DATA_DIR / "data_station1_2023.csv")
test_matrix = pd.read_csv(DATA_DIR / "test_matrix.csv") 

y_test = test_data['co'].values
X_test_raw = test_matrix.copy()

# --- THE FIX: Slice the matrix to exactly match the length of the test targets ---

X_test_raw = X_test_raw.drop(columns=['Day_365'])

# Drop one hour dummy
X_test_raw = X_test_raw.drop(columns=['Hour_24'])

# ── Step 2: Align Test Features with Training Features ──
# Ensure the test matrix has the exact same columns as the training matrix 'X'


# .reindex() matches the columns perfectly to the training set.
# Missing columns are filled with 0, and extra columns not in the training set are dropped.
X_test = X_test_raw.reindex(columns=X.columns, fill_value=0)
X_test_vals = X_test.values

# ── Step 3: Base Model (OLS) Predictions ──
# Using the 'beta' array calculated during training
y_hat_test = X_test_vals @ beta

# ── Step 4: Calculate Standard Results for Base Model ──
test_residuals_base = y_test - y_hat_test
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.plot(y_test, label='Actual')
plt.plot(y_hat_test, label='OLS')
plt.legend()
plt.show()
rmse_base = np.sqrt(mean_squared_error(y_test, y_hat_test))
mae_base = mean_absolute_error(y_test, y_hat_test)
r2_base = 1 - (np.var(test_residuals_base) / np.var(y_test))

print("\n" + "="*40)
print("--- BASE MODEL (OLS) TEST RESULTS ---")
print(f"Test RMSE: {rmse_base:.4f}")
print(f"Test MAE:  {mae_base:.4f}")
print(f"Test R²:   {r2_base:.4f}")
print("="*40)

# ── Step 5: Combined Model Forecast (OLS + ARMA) 1-Step Ahead ──
# ============================================================
# STEP 5 : ONE-STEP-AHEAD ARMA (ONLINE UPDATING)
# ============================================================

print("\nEvaluating ARMA using one-step-ahead updating...")

# Base model residuals (actual - regression)
test_residuals_base = y_test - y_hat_test

# Initialize the ARMA state using the trained model
test_arma = arma_fit.apply(
    test_residuals_base,
    refit=False
)

# One-step-ahead residual predictions
arma_1step = test_arma.fittedvalues

# Final prediction
combined_forecast_test = y_hat_test + arma_1step

# Errors
combined_residuals = y_test - combined_forecast_test

rmse_combined = np.sqrt(mean_squared_error(y_test,
                                           combined_forecast_test))

mae_combined = mean_absolute_error(y_test,
                                   combined_forecast_test)

r2_combined = 1 - np.var(combined_residuals)/np.var(y_test)

print("\n==============================")
print("OLS + ARMA (One-Step Ahead)")
print("==============================")
print(f"RMSE : {rmse_combined:.4f}")
print(f"MAE  : {mae_combined:.4f}")
print(f"R²   : {r2_combined:.4f}")

test_data["OLS"] = y_hat_test
test_data["ARMA"] = arma_1step
test_data["Combined"] = combined_forecast_test

""
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

# If your test data has a datetime column
if 'datetime' in test_data.columns:
    plt.plot(test_data['datetime'], y_test,
             label='Actual CO', linewidth=2)
    plt.plot(test_data['datetime'], combined_forecast_test,
             label='OLS + ARMA Prediction', linewidth=2)
    plt.xlabel("Time")
else:
    # Otherwise plot against sample index
    plt.plot(y_test, label='Actual CO', linewidth=2)
    plt.plot(combined_forecast_test,
             label='OLS + ARMA Prediction', linewidth=2)
    plt.xlabel("Sample")

plt.ylabel("CO Concentration")
plt.title("Actual vs Combined Forecast (Test Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()