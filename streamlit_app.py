import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA

# Load data (assumes cbr_data.csv is preprocessed and available)
cbr_data = pd.read_csv("cbr2012.csv")

# Streamlit sidebar inputs
st.sidebar.title("Model Options")
model_choice = st.sidebar.selectbox("Select model:", ["Linear", "Quadratic", "ARIMA"], index=0)
train_start = st.sidebar.selectbox("Training start year:", list(range(2012, 2020 - 4)))
train_end = st.sidebar.selectbox("Training end year:", list(range(train_start + 4, 2020)), index=(2019 - train_start - 4))

st.title("CBR Trends and Excess CBRs by Country")

modate_1 = 729  # Oct 2020
modate_2 = 753  # Oct 2022

# Function to generate model-based trend and CI
def get_predictions(df, model_type):
    result = {}
    df_train = df[(df['year'] >= train_start) & (df['year'] <= train_end)].copy()
    df_full = df.copy()

    if df_train.empty:
        return None

    try:
        if model_type == "Linear":
            X_train = sm.add_constant(df_train['modate'].astype(float))
            y_train = df_train['CBR'].astype(float)
            model = sm.OLS(y_train, X_train).fit()
            X_full = sm.add_constant(df_full['modate'].astype(float))
            pred = model.get_prediction(X_full).summary_frame(alpha=0.05)

        elif model_type == "Quadratic":
            df_train['modate_squared'] = df_train['modate'] ** 2
            df_full['modate_squared'] = df_full['modate'] ** 2
            model = smf.ols('CBR ~ modate + modate_squared', data=df_train).fit()
            pred = model.get_prediction(df_full[['modate', 'modate_squared']]).summary_frame(alpha=0.05)

        elif model_type == "ARIMA":
            y_train = df_train['CBR'].astype(float)
            model = ARIMA(y_train, order=(1, 1, 1)).fit()
            forecast = model.get_forecast(steps=len(df_full))
            pred = forecast.summary_frame(alpha=0.05)
            pred.index = df_full.index  # align index

        return pred

    except Exception as e:
        return None

# Store predictions for all countries first
cbr_data['prediction'] = np.nan
countries = cbr_data['country'].unique()
for country in countries:
    df_country = cbr_data[cbr_data['country'] == country].copy()
    pred = get_predictions(df_country, model_choice)
    if pred is not None:
        cbr_data.loc[df_country.index, 'prediction'] = pred['mean'].values

# Compute excess CBR
cbr_data['excess_cbr'] = cbr_data['CBR'] - cbr_data['prediction']

# === Dispersion plot FIRST ===
st.header("Global Excess CBR Scatterplot")
fig, ax = plt.subplots(figsize=(12, 6))
for country in countries:
    df = cbr_data[cbr_data['country'] == country]
    ax.scatter(df['modate'], df['excess_cbr'], s=10, alpha=0.6)
ax.axvline(modate_1, color='red', linestyle='--', label='Oct 2020')
ax.axvline(modate_2, color='blue', linestyle='--', label='Oct 2022')
ax.set_title(f'Excess CBRs — {model_choice} Model')
ax.set_xlabel('Modate')
ax.set_ylabel('Excess CBR')
ax.grid(True)
st.pyplot(fig)

# Plotting for each country
countries = cbr_data['country'].unique()
for country in countries:
    df_country = cbr_data[cbr_data['country'] == country].copy()
    pred = get_predictions(df_country, model_choice)

    if pred is not None:
        modate = df_country['modate'].astype(float).to_numpy()
        actual = df_country['CBR'].astype(float).to_numpy()
        mean = pred['mean'].to_numpy()
        lower = pred['mean_ci_lower'].to_numpy()
        upper = pred['mean_ci_upper'].to_numpy()

        # Plot for countries
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(modate, actual, color='black', s=15, label='Actual CBR')
        ax.plot(modate, mean, color='blue', linestyle='--', label=f'{model_choice} Trend')
        ax.fill_between(modate, lower, upper, color='blue', alpha=0.2, label='95% CI')
        ax.axvline(modate_1, color='red', linestyle='--')
        ax.axvline(modate_2, color='blue', linestyle='--')
        ax.set_title(f'{country} — {model_choice} Trend')
        ax.set_xlabel('Modate')
        ax.set_ylabel('CBR')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

