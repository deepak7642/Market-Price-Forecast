import streamlit as st
import pandas as pd
from prophet import Prophet
import holidays
import logging
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress logging output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Load data
df_ts = pd.read_csv('Kerala data.csv')
future_weather_data_7 = pd.read_csv('future data 7.csv')

# Data Preprocessing
df_ts['datetime'] = pd.to_datetime(df_ts['datetime'], format='%d-%m-%Y')
future_weather_data_7['datetime'] = pd.to_datetime(future_weather_data_7['datetime'], format='%d-%m-%Y')

# Function to get the season
def get_season(month):
    if month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    elif month in [12, 1, 2]: return 'Winter'
    elif month in [10, 11, 12, 1]: return 'Rabi'
    elif month in [5, 6, 7, 8]: return 'Kharif'
    else: return 'Whole Year'

# Add season and date-related columns
df_ts['Season'] = df_ts['datetime'].dt.month.apply(get_season)
df_ts['year'] = df_ts['datetime'].dt.year
df_ts['month'] = df_ts['datetime'].dt.month
df_ts['day'] = df_ts['datetime'].dt.day
df_ts['weekday'] = df_ts['datetime'].dt.weekday
df_ts['week_of_year'] = df_ts['datetime'].dt.isocalendar().week

# Add holidays
start_year = 2010
end_year = 2010
all_holidays = {}
for year in range(start_year, end_year + 1):
    india_holidays = holidays.India(years=year)
    all_holidays[year] = india_holidays

df_ts['Holiday'] = 'No Holiday'
for year, holiday_list in all_holidays.items():
    for date in holiday_list:
        df_ts.loc[df_ts['datetime'].dt.date == date, 'Holiday'] = 'Holiday'

df_ts['min_price'].fillna(method='ffill', inplace=True)
df_ts['max_price'].fillna(method='ffill', inplace=True)
df_ts['modal_price'].fillna(method='ffill', inplace=True)

label_encoder = LabelEncoder()
df_ts['Season'] = label_encoder.fit_transform(df_ts['Season'])
df_ts['Holiday'] = label_encoder.fit_transform(df_ts['Holiday'])

future_weather_data_7['Season'] = future_weather_data_7['datetime'].dt.month.apply(get_season)
future_weather_data_7['year'] = future_weather_data_7['datetime'].dt.year
future_weather_data_7['month'] = future_weather_data_7['datetime'].dt.month
future_weather_data_7['day'] = future_weather_data_7['datetime'].dt.day
future_weather_data_7['weekday'] = future_weather_data_7['datetime'].dt.weekday
future_weather_data_7['week_of_year'] = future_weather_data_7['datetime'].dt.isocalendar().week

future_weather_data_7['Holiday'] = 'No Holiday'
for year, holiday_list in all_holidays.items():
    for date in holiday_list:
        future_weather_data_7.loc[future_weather_data_7['datetime'].dt.date == date, 'Holiday'] = 'Holiday'

future_weather_data_7['Season'] = label_encoder.fit_transform(future_weather_data_7['Season'])
future_weather_data_7['Holiday'] = label_encoder.fit_transform(future_weather_data_7['Holiday'])

# Columns of interest for forecasting
cols = ['min_price', 'max_price', 'modal_price']
regressors = ['tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'precip', 'windspeed', 'solarradiation', 'uvindex', 'Season', 'Holiday']

combined_forecast_df = pd.DataFrame()

grouped_df = df_ts.groupby(['district_name', 'market_name', 'commodity_name', 'variety'])

for (district, market, commodity, variety), group_df in grouped_df:
    results = []
    for col in cols:
        subdf = group_df[['datetime', col] + regressors].rename(columns={"datetime": "ds", col: "y"})
        m = Prophet()
        for regressor in regressors:
            m.add_regressor(regressor)
        m.fit(subdf)
        future = m.make_future_dataframe(periods=7)
        for regressor in regressors:
            future[regressor] = pd.concat([subdf[regressor], future_weather_data_7[regressor].iloc[:7]]).reset_index(drop=True).iloc[:future.shape[0]]
        forecast = m.predict(future)
        forecast = future[['ds'] + regressors].merge(forecast[['ds', 'yhat']], on='ds', how='left')
        forecast = forecast.rename(columns={'yhat': f'forecast_{col}'})
        results.append(forecast)

    df_group_predict = group_df[['datetime']].rename(columns={'datetime': 'ds'})
    for result in results:
        df_group_predict = pd.merge(df_group_predict, result, on='ds', how='outer')

    df_group_predict['District Name'] = district
    df_group_predict['Market Name'] = market
    df_group_predict['Commodity'] = commodity
    df_group_predict['Variety'] = variety

    column_order = [col for col in df_group_predict.columns if col not in ['forecast_min_price', 'forecast_max_price', 'forecast_modal_price']] + ['forecast_min_price', 'forecast_max_price', 'forecast_modal_price']
    df_group_predict = df_group_predict.reindex(columns=column_order)

    columns_to_drop = ['tempmax_x', 'tempmin_x', 'temp_x', 'dew_x', 'humidity_x', 'precip_x', 'windspeed_x', 'solarradiation_x', 'uvindex_x', 'Season_x', 'Holiday_x', 'tempmax_y', 'tempmin_y', 'temp_y', 'dew_y', 'humidity_y', 'precip_y', 'windspeed_y', 'solarradiation_y', 'uvindex_y', 'Season_y', 'Holiday_y', 'tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'precip', 'windspeed', 'solarradiation', 'uvindex', 'Season', 'Holiday']
    df_group_predict.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    df_group_predict.rename(columns={'ds': 'datetime', 'forecast_min_price': 'forecast_min_price', 'forecast_max_price': 'forecast_max_price', 'forecast_modal_price': 'forecast_modal_price'}, inplace=True)

    combined_forecast_df = pd.concat([combined_forecast_df, df_group_predict], ignore_index=True)

# Get unique values for dropdowns
districts = df_ts['district_name'].unique()
markets = df_ts['market_name'].unique()
commodities = df_ts['commodity_name'].unique()
varieties = df_ts['variety'].unique()

# Streamlit interface
st.title('Price Forecasting App')

district = st.selectbox("Select District Name", districts)
market = st.selectbox("Select Market Name", markets)
commodity = st.selectbox("Select Commodity", commodities)
variety = st.selectbox("Select Variety", varieties)
start_date = st.date_input('Select start date', min_value=df_ts['datetime'].min(), max_value=df_ts['datetime'].max(), value=df_ts['datetime'].min())

if st.button('Generate Forecast'):
    try:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = start_datetime + pd.DateOffset(days=7)

        filtered_df = combined_forecast_df[
            (combined_forecast_df['District Name'] == district) &
            (combined_forecast_df['Market Name'] == market) &
            (combined_forecast_df['Commodity'] == commodity) &
            (combined_forecast_df['Variety'] == variety) &
            (combined_forecast_df['datetime'] >= start_datetime) &
            (combined_forecast_df['datetime'] < end_datetime)
        ]

        if filtered_df.empty:
            st.write("No data found for the specified input.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['forecast_min_price'], mode='lines+markers', name='Forecast Min Price'))
            fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['forecast_max_price'], mode='lines+markers', name='Forecast Max Price'))
            fig.add_trace(go.Scatter(x=filtered_df['datetime'], y=filtered_df['forecast_modal_price'], mode='lines+markers', name='Forecast Modal Price'))

            fig.update_layout(title=f'Forecasted Prices for {commodity} ({variety}) in {market}, {district}', xaxis_title='Date', yaxis_title='Price', template='plotly_white', width=1000, height=500)

            st.plotly_chart(fig)

    except ValueError:
        st.write("Invalid date format. Please enter the date in YYYY-MM-DD format.")
