import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from pygam import LinearGAM, s
import plotly.express as px
import plotly.graph_objects as go

# Patch compatibility for NumPy and SciPy (deprecation fix)
import patch_pygam

# Load data (assumes cbr_data.csv is preprocessed and available)
cbr_data = pd.read_csv("cbr2012.csv")

# Streamlit sidebar inputs
st.sidebar.title("Model Options")
model_choice = st.sidebar.selectbox("Select model:", ["Linear", "Quadratic", "ARIMA", "GAM"], index=0)
train_start = st.sidebar.selectbox("Training start year:", list(range(2012, 2022 - 4)))
year_labels = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020 (during Covid-19)", "2021 (during Covid-19)"]
year_values = list(range(2016, 2022))
train_end = st.sidebar.selectbox("Training end year:", year_labels, index=7)
train_end_val = 2012 + year_labels.index(train_end)

# Sidebar Y-axis range selector
y_range = st.sidebar.selectbox("Y-axis range for excess CBR", options=[0.005, 0.01], index=0)

st.title("CBR Trends and Excess CBRs by Country")

modate_1 = 729  # Oct 2020
modate_2 = 753  # Oct 2022

if model_choice == "ARIMA":
    st.info("ARIMA forecasts are shown only after the training period ends. No training fit is plotted.")

# Store predictions globally for dispersion
cbr_data['prediction'] = np.nan
countries = sorted(cbr_data['country'].unique())

# Function to generate model-based trend and CI
def get_predictions(df, model_type):
    df_train = df[(df['year'] >= train_start) & (df['year'] <= train_end_val)].copy()
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
            return pred

        elif model_type == "Quadratic":
            df_train['modate_squared'] = df_train['modate'] ** 2
            df_full['modate_squared'] = df_full['modate'] ** 2
            model = smf.ols('CBR ~ modate + modate_squared', data=df_train).fit()
            pred = model.get_prediction(df_full[['modate', 'modate_squared']]).summary_frame(alpha=0.05)
            return pred

        elif model_type == "ARIMA":
            y_train = df_train['CBR'].astype(float)
            model = ARIMA(y_train, order=(1, 1, 1)).fit()
            steps = len(df_full) - len(df_train)
            if steps <= 0:
                return None
            forecast = model.get_forecast(steps=steps).summary_frame(alpha=0.05)
            pred = pd.DataFrame(index=df_full.index, columns=['mean', 'mean_ci_lower', 'mean_ci_upper'])
            pred.iloc[-steps:] = forecast[['mean', 'mean_ci_lower', 'mean_ci_upper']].values
            return pred

        elif model_type == "GAM":
            X_train = df_train['modate'].values.reshape(-1, 1)
            y_train = df_train['CBR'].values
            gam = LinearGAM(s(0))
            try:
                gam.fit(X_train, y_train)
                X_full = df_full['modate'].values.reshape(-1, 1)
                mean = gam.predict(X_full)
                ci = gam.prediction_intervals(X_full, width=0.95)
                pred = pd.DataFrame({
                    'mean': mean,
                    'mean_ci_lower': ci[:, 0],
                    'mean_ci_upper': ci[:, 1]
                }, index=df_full.index)
                return pred
            except Exception as e:
                st.warning(f"GAM model failed to fit for a country: {e}")
                return None

    except Exception as e:
        st.warning(f"Prediction error for model {model_type}: {e}")
        return None

for country in countries:
    df_country = cbr_data[cbr_data['country'] == country].copy()
    pred = get_predictions(df_country, model_choice)
    if pred is not None:
        cbr_data.loc[df_country.index, 'prediction'] = pred['mean'].values

cbr_data['excess_cbr'] = cbr_data['CBR'] - cbr_data['prediction']

# === Interactive Dispersion plot with Plotly ===
st.header("Global Excess CBR Scatterplot")
dispersion_data = cbr_data[['modate', 'country', 'excess_cbr']].dropna()

selected_country = st.selectbox("Highlight a specific country:", options=["None"] + countries)

if selected_country != "None":
    dispersion_data['color'] = dispersion_data['country'].apply(lambda c: "highlight" if c == selected_country else "other")
    dispersion_data['sort_order'] = dispersion_data['color'].apply(lambda x: 0 if x == 'other' else 1)
    dispersion_data = dispersion_data.sort_values('sort_order')
    color_map = {"highlight": "red", "other": "lightgray"}
    fig_disp = px.scatter(
        dispersion_data,
        x='modate',
        y='excess_cbr',
        color='color',
        color_discrete_map=color_map,
        hover_name='country',
        labels={'modate': 'Modate', 'excess_cbr': 'Excess CBR'},
        title=f'Excess CBRs — {model_choice} Model (highlight: {selected_country})'
    )
    fig_disp.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.6)
else:
    fig_disp = px.scatter(
        dispersion_data,
        x='modate',
        y='excess_cbr',
        color='country',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_name='country',
        labels={'modate': 'Modate', 'excess_cbr': 'Excess CBR'},
        title=f'Excess CBRs — {model_choice} Model'
    )

fig_disp.add_vline(x=modate_1, line_dash="dash", line_color="red")
fig_disp.add_annotation(x=modate_1, y=0.011, text="Oct 2020\n(9mo post-Covid)", showarrow=False, font=dict(color="red"))
fig_disp.add_vline(x=modate_2, line_dash="dash", line_color="blue")
fig_disp.add_annotation(x=modate_2, y=0.011, text="Oct 2022\n(9mo post-Ukraine)", showarrow=False, font=dict(color="blue"))
fig_disp.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
fig_disp.update_layout(showlegend=False)
fig_disp.update_yaxes(range=[-y_range, y_range])

if y_range == 0.005:
    if y_range == 0.005 and (
        (dispersion_data['excess_cbr'] > 0.005).any() or (dispersion_data['excess_cbr'] < -0.005).any()
    ):
        st.warning("Some points exceed ±0.005 and may be clipped. Adjust the y-axis range in the sidebar to display the full series.")

st.plotly_chart(fig_disp, use_container_width=True)

# === Country-wise plots ===
if selected_country != "None":
    ordered_countries = [selected_country] + [c for c in countries if c != selected_country]
else:
    ordered_countries = countries

for country in ordered_countries:
    df_country = cbr_data[cbr_data['country'] == country].copy()
    pred = get_predictions(df_country, model_choice)

    if pred is not None:
        modate = df_country['modate'].astype(float).to_numpy()
        actual = df_country['CBR'].astype(float).to_numpy()
        mean = pred['mean'].astype(float).to_numpy()
        lower = pred['mean_ci_lower'].astype(float).to_numpy()
        upper = pred['mean_ci_upper'].astype(float).to_numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(modate, actual, color='black', s=15, label='Actual CBR')

        if model_choice == "ARIMA":
            forecast_mask = ~np.isnan(mean)
            ax.plot(modate[forecast_mask], mean[forecast_mask], color='purple', linestyle='--', label='ARIMA Forecast')
            ax.fill_between(modate[forecast_mask], lower[forecast_mask], upper[forecast_mask], color='purple', alpha=0.2, label='95% CI')
            ax.axvline(df_country[df_country['year'] == train_end_val]['modate'].min(), color='gray', linestyle=':', label='Training End')
        else:
            ax.plot(modate, mean, color='blue', linestyle='--', label=f'{model_choice} Trend')
            ax.fill_between(modate, lower, upper, color='blue', alpha=0.2, label='95% CI')

        ax.axvline(modate_1, color='red', linestyle='--', label='Oct 2020')
        ax.axvline(modate_2, color='blue', linestyle='--', label='Oct 2022')
        ax.set_title(f'{country} — {model_choice} Trend')
        ax.set_xlabel('Modate')
        ax.set_ylabel('CBR')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
