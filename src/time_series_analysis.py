import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import seaborn as sns

# 1. Load Data and Setup Index
from pathlib import Path

# This file is stored in src/, so the data folder is one level above it.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

file_path = DATA_DIR / "data_station1.csv"
df = pd.read_csv(file_path)

# Generate datetime index and assign it to the dataframe
dt_index = pd.date_range(start="2010-01-01 00:00:00", end="2023-03-31 23:00:00", freq="h")

# Ensure the generated index matches the length of your dataframe
# (If it doesn't, you may need to adjust the date range or truncate the data)
df['datetime'] = dt_index[:len(df)] 
df = df.set_index("datetime")

# 2. Extract raw time counts
df['day'] = df.index.dayofyear
df['hour'] = df.index.hour
df['month'] = df.index.month

# 3. Convert to radians
df['day_rad'] = 2 * np.pi * df['day'] / 365
df['hour_rad'] = 2 * np.pi * df['hour'] / 24

# 4. Calculate sine and cosine feature columns
df['cos_annual'] = np.cos(df['day_rad'])
df['sin_annual'] = np.sin(df['day_rad'])
df['cos_daily'] = np.cos(df['hour_rad'])

# 5. Plotting the Time Series
sco = df['co']
spm_2p5 = df['pm2.5']
snox = df['nox']
sso2 = df['so2']
so3 = df['o3']

fig, axes = plt.subplots(5, 1, figsize=(14,10), sharex=True)
axes[0].plot(sco)
axes[0].set_title("CO")
axes[1].plot(spm_2p5)
axes[1].set_title("PM2.5")
axes[2].plot(snox)
axes[2].set_title("NOx")
axes[3].plot(sso2)
axes[3].set_title("SO2")
axes[4].plot(so3)
axes[4].set_title("O3")
plt.tight_layout()
plt.show()

# 6. ADF Test Function
def adf_test(series, title=''):
    """Pass in a time series and an optional title, returns an ADF report"""
    print(f'Augmented Dickey-Fuller Test: {title}')
    
    result = adfuller(series.dropna(), autolag='AIC') 
    labels = ['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'Critical Value ({key})'] = val
        
    print(out)
    
    # Interpretation
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is STATIONARY")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is NON-STATIONARY")
    print("-" * 30)

# Run ADF test on CO
adf_test(sco, title='CO Levels')

# 7. Seasonal Decomposition
# Note: period=24 for daily seasonality on hourly data. 
# Use period=8760 for annual seasonality on hourly data.
result = seasonal_decompose(sco.dropna(), model='additive', period=24)
result.plot()
plt.show()

# 8. ACF and PACF Plots
series = sco.dropna().diff().dropna()
fig, ax = plt.subplots(1,2, figsize=(12,4))
plot_acf(series, lags=80, ax=ax[0])
plot_pacf(series, lags=80, ax=ax[1])
plt.show()

# 9. Kruskal-Wallis Test for Seasonality
monthly_groups = []
for m in range(1, 13):
    month_data = df[df['month'] == m]['co'].dropna()
    monthly_groups.append(month_data)

stat, p_value = stats.kruskal(*monthly_groups)

print(f"Kruskal-Wallis Statistic: {stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("Result: Reject Null Hypothesis.")
    print("Conclusion: STRONG SEASONALITY DETECTED (Months differ significantly).")
else:
    print("Result: Fail to reject Null Hypothesis.")
    print("Conclusion: No seasonality detected.")

# 10. Boxplot for Seasonality Check
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='co', data=df)
plt.title('Seasonality Check: CO Distribution by Month')
plt.show()