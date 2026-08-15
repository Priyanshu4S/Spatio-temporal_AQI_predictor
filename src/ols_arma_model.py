
# -*- coding: utf-8 -*-
"""
OLS + ARMA Forecasting (True Out-of-Sample)

Changes from previous version
-----------------------------
1. Uses design matrix with an existing intercept column.
2. Drops one Day and one Hour dummy to avoid the dummy-variable trap.
3. Uses np.linalg.solve instead of explicitly inverting X'X.
4. Uses ARMA.forecast() on the test set (NO data leakage).
5. Reports train/test RMSE, MAE and R².
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------------

from pathlib import Path

# This file is stored in src/, so the data folder is one level above it.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train_data_path   = DATA_DIR / "2019-2022_co_ds1.csv"
train_matrix_path = DATA_DIR / "Design_matrix_35040x390.csv"

test_data_path    = DATA_DIR / "data_station1_2023.csv"
test_matrix_path  = DATA_DIR / "test_matrix.csv"

# ------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------

data = pd.read_csv(train_data_path)
X = pd.read_csv(train_matrix_path)

# Drop same reference dummies everywhere
for c in ["Day_365", "Hour_24"]:
    if c in X.columns:
        X = X.drop(columns=c)

y = data["co"].to_numpy()

rank = np.linalg.matrix_rank(X.values)
print("Training shape :", X.shape)
print("Training rank  :", rank)

if rank != X.shape[1]:
    raise ValueError("Training matrix is still rank deficient.")

XtX = X.values.T @ X.values
Xty = X.values.T @ y

beta = np.linalg.solve(XtX, Xty)

y_hat = X.values @ beta
eps = y - y_hat

train_rmse = np.sqrt(mean_squared_error(y, y_hat))
train_mae = mean_absolute_error(y, y_hat)
train_r2 = 1 - np.var(eps)/np.var(y)

print("\nTRAIN")
print("RMSE :", round(train_rmse,4))
print("MAE  :", round(train_mae,4))
print("R²   :", round(train_r2,4))

# Statsmodels verification
model = sm.OLS(y, X).fit()
print("\nStatsmodels R² :", round(model.rsquared,4))

# ------------------------------------------------------------------
# ARMA
# ------------------------------------------------------------------

arma = ARIMA(eps, order=(1,0,1))
arma_fit = arma.fit()

combined_train = y_hat + arma_fit.fittedvalues
train_res = y - combined_train

train_r2_comb = 1 - np.var(train_res)/np.var(y)

print("\nCombined Train R² :", round(train_r2_comb,4))

# ------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------

test = pd.read_csv(test_data_path)
Xtest = pd.read_csv(test_matrix_path)

for c in ["Day_365", "Hour_24"]:
    if c in Xtest.columns:
        Xtest = Xtest.drop(columns=c)

Xtest = Xtest.reindex(columns=X.columns)

y_test = test["co"].to_numpy()

y_hat_test = Xtest.values @ beta

base_rmse = np.sqrt(mean_squared_error(y_test,y_hat_test))
base_mae = mean_absolute_error(y_test,y_hat_test)
base_r2 = 1 - np.var(y_test-y_hat_test)/np.var(y_test)

print("\nTEST (OLS)")
print("RMSE :", round(base_rmse,4))
print("MAE  :", round(base_mae,4))
print("R²   :", round(base_r2,4))

# ------------------------------------------------------------------
# TRUE FORECAST
# ------------------------------------------------------------------

residual_forecast = arma_fit.forecast(steps=len(y_test))

combined_test = y_hat_test + residual_forecast

comb_rmse = np.sqrt(mean_squared_error(y_test,combined_test))
comb_mae = mean_absolute_error(y_test,combined_test)
comb_r2 = 1 - np.var(y_test-combined_test)/np.var(y_test)

print("\nTEST (OLS + ARMA Forecast)")
print("RMSE :", round(comb_rmse,4))
print("MAE  :", round(comb_mae,4))
print("R²   :", round(comb_r2,4))

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

test["OLS"] = y_hat_test
test["ARMA_forecast"] = residual_forecast
test["Combined"] = combined_test

# ------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------

plt.figure(figsize=(14,5))
plt.plot(y,label="Actual")
plt.plot(y_hat,label="OLS")
plt.legend()
plt.title("Training : Actual vs OLS")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(14,5))
plt.plot(y,label="Actual")
plt.plot(combined_train,label="OLS + ARMA")
plt.legend()
plt.title("Training : Combined")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(14,5))
plt.plot(y_test,label="Actual")
plt.plot(y_hat_test,label="OLS")
plt.legend()
plt.title("Test : Actual vs OLS")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(14,5))
plt.plot(y_test,label="Actual")
plt.plot(combined_test,label="OLS + ARMA Forecast")
plt.legend()
plt.title("Test : True Forecast")
plt.grid()
plt.tight_layout()

plt.show()

beta_series = pd.Series(beta,index=X.columns)

day_beta = beta_series[beta_series.index.str.startswith("Day_")]
hour_beta = beta_series[beta_series.index.str.startswith("Hour_")]

plt.figure(figsize=(14,5))
plt.plot([int(i.split("_")[1]) for i in day_beta.index],day_beta.values)
plt.title("Day Coefficients")
plt.grid()

plt.figure(figsize=(8,4))
plt.plot([int(i.split("_")[1]) for i in hour_beta.index],hour_beta.values,marker="o")
plt.title("Hour Coefficients")
plt.grid()

plt.show()
