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
model_choice = st.sidebar.selectbox(
    "Select model:",
    ["Linear", "Quadratic", "Cubic", "ARIMA", "GAM", "Best fitting OLS"],
    index=0
)
train_start = st.sidebar.selectbox("Training start year:", list(range(2012, 2022 - 4)), index=3)
year_labels = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020 (during Covid-19)", "2021 (during Covid-19)"]
year_values = list(range(2016, 2022))
train_end = st.sidebar.selectbox("Training end year:", year_labels, index=7)
train_end_val = 2012 + year_labels.index(train_end)

# Sidebar Y-axis range selector
y_range = st.sidebar.selectbox("Y-axis range for excess CBR", options=[0.005, 0.01], index=0)
display_regions = st.sidebar.checkbox("Display regions", value=False)

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

        elif model_type == "Cubic":
            df_train['modate_squared'] = df_train['modate'] ** 2
            df_train['modate_cubed'] = df_train['modate'] ** 3
            df_full['modate_squared'] = df_full['modate'] ** 2
            df_full['modate_cubed'] = df_full['modate'] ** 3
            model = smf.ols('CBR ~ modate + modate_squared + modate_cubed', data=df_train).fit()
            pred = model.get_prediction(df_full[['modate', 'modate_squared', 'modate_cubed']]).summary_frame(alpha=0.05)
            return pred

        elif model_type == "Best fitting OLS":
            best_model = None
            best_error = float('inf')
            best_pred = None

            # Test Linear model
            try:
                pred_linear = get_predictions(df, "Linear")
                error_linear = ((df_train['CBR'] - pred_linear['mean']) ** 2).mean()
                if error_linear < best_error:
                    best_model = "Linear"
                    best_error = error_linear
                    best_pred = pred_linear
            except Exception:
                pass

            # Test Quadratic model
            try:
                pred_quadratic = get_predictions(df, "Quadratic")
                error_quadratic = ((df_train['CBR'] - pred_quadratic['mean']) ** 2).mean()
                if error_quadratic < best_error:
                    best_model = "Quadratic"
                    best_error = error_quadratic
                    best_pred = pred_quadratic
            except Exception:
                pass

            # Test Cubic model
            try:
                pred_cubic = get_predictions(df, "Cubic")
                error_cubic = ((df_train['CBR'] - pred_cubic['mean']) ** 2).mean()
                if error_cubic < best_error:
                    best_model = "Cubic"
                    best_error = error_cubic
                    best_pred = pred_cubic
            except Exception:
                pass

            # Store the best model name in the dataframe
            df['best_model'] = best_model
            return best_pred

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

best_models = {}
for country in countries:
    df_country = cbr_data[cbr_data['country'] == country].copy()
    pred = get_predictions(df_country, model_choice)
    if pred is not None:
        cbr_data.loc[df_country.index, 'prediction'] = pred['mean'].values
        best_models[country] = df_country['best_model'].iloc[0] if 'best_model' in df_country else model_choice

cbr_data['excess_cbr'] = cbr_data['CBR'] - cbr_data['prediction']

# === Interactive Dispersion plot with Plotly ===
st.header("Global Excess CBR Scatterplot")
region_map = {
    'Austria': 'Western Europe', 'Belgium': 'Western Europe', 'France': 'Western Europe', 'Germany': 'Western Europe',
    'Ireland': 'Western Europe', 'Luxembourg': 'Western Europe', 'Netherlands': 'Western Europe', 'Switzerland': 'Western Europe', 'United Kingdom': 'Western Europe',
    'Greece': 'Southern Europe', 'Italy': 'Southern Europe', 'Portugal': 'Southern Europe', 'Spain': 'Southern Europe',
    'Canada': 'North America', 'United States of America': 'North America',
    'Bulgaria': 'Former Soviet Bloc', 'Croatia': 'Former Soviet Bloc', 'Czech Republic': 'Former Soviet Bloc', 'Estonia': 'Former Soviet Bloc',
    'Hungary': 'Former Soviet Bloc', 'Latvia': 'Former Soviet Bloc', 'Lithuania': 'Former Soviet Bloc', 'Poland': 'Former Soviet Bloc',
    'Romania': 'Former Soviet Bloc', 'Russian Federation': 'Former Soviet Bloc', 'Serbia': 'Former Soviet Bloc', 'Slovakia': 'Former Soviet Bloc', 'Slovenia': 'Former Soviet Bloc',
    'Australia': 'Other High-Income', 'New Zealand': 'Other High-Income', 'Israel': 'Other High-Income', 'Japan': 'Asia', 'Korea': 'Asia',
    'Iceland': 'Northern Europe', 'Finland': 'Northern Europe', 'Norway': 'Northern Europe', 'Sweden': 'Northern Europe', 'Denmark': 'Northern Europe'
}
dispersion_data = cbr_data[['modate', 'country', 'excess_cbr']].dropna()
# Keep region_map for future grouping or tooltips, but keep country as primary color
region_group_map = dispersion_data['country'].map(region_map)
# Optional: could use this for grouping/filtering if needed

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
    fig_disp.add_hline(y=0, line_dash='dot', line_color='darkgray', opacity=1)
else:
    color_choice = 'region_group' if display_regions else 'country'
    color_seq = px.colors.qualitative.Pastel
    dispersion_data['region_group'] = dispersion_data['country'].map(region_map)
    fig_disp = px.scatter(
        dispersion_data,
        x='modate',
        y='excess_cbr',
        color=color_choice,
        color_discrete_sequence=color_seq,
        hover_name='country',
        labels={'modate': 'Modate', 'excess_cbr': 'Excess CBR'},
        title=f'Excess CBRs — {model_choice} Model'
    )
    if display_regions:
        fig_disp.update_xaxes(tickformat='%Y-%b')
        fig_disp.add_hline(y=0, line_dash='dot', line_color='darkgray', opacity=0.5)
    fig_disp.add_hline(y=0, line_dash='dot', line_color='darkgray', opacity=0.5)

fig_disp.add_vline(
    x=modate_1,
    line_dash="dash",
    line_color="black",
    line_width=0.8,
    annotation_text="Oct 2020 (9mo post-Covid)",
    annotation_position="top",
    annotation_font=dict(color="black", size=10)
)

fig_disp.add_vline(
    x=modate_2,
    line_dash="dash",
    line_color="black",
    line_width=0.8,
    annotation_text="Oct 2022 (9mo post-Ukraine)",
    annotation_position="top right",
    annotation_font=dict(color="black", size=10)
)
fig_disp.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
fig_disp.update_layout(showlegend=display_regions)
modate_min = cbr_data['modate'].min()
modate_max = cbr_data['modate'].max()
fig_disp.update_xaxes(range=[modate_min, modate_max])
fig_disp.update_yaxes(range=[-y_range, y_range])
if display_regions:
    fig_disp.update_xaxes(tickformat='%Y-%b')
    fig_disp.add_hline(y=0, line_dash='dot', line_color='darkgray', opacity=0.5)
modate_min = cbr_data['modate'].min()
modate_max = cbr_data['modate'].max()
fig_disp.update_xaxes(range=[modate_min, modate_max], matches='x')


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
        actual_model = best_models.get(country, model_choice)
        ax.set_title(f'{country} — {actual_model} Trend')

        ax.set_xlabel('Modate')
        
        ax.set_ylabel('CBR')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
