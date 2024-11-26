from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from io import StringIO
from flask_cors import CORS

# Load both trained models
with open("model_IQ.pkl", "rb") as file:
    model_IQ = pickle.load(file)

with open("model_SJ.pkl", "rb") as file:
    model_SJ = pickle.load(file)

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

def process_weather_data(city):
    response = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?unitGroup=metric&include=days&key=TYAMJ9DT4BBGRX76CPR4FQ7VA&contentType=csv")
    if response.status_code != 200:
        return None, {"error": f"Weather API error for {city}", "status_code": response.status_code}
    
    data = StringIO(response.text)
    weather_df = pd.read_csv(data)

    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df['week_of_year'] = weather_df['datetime'].dt.isocalendar().week
    weather_df['year'] = weather_df['datetime'].dt.year
    
    drop_columns = ['name', 'datetime','feelslikemax','feellikemin','feelslike','precipprob',
                    'preciptype','snow','snowdepth','windgust','winddir','sealevelpressure','cloudcover','visibility',
                    'solarradiation','solarenergy','uvindex','severerisk','conditions','icon','stations',
                    'sunrise','feelslikemin', 'precipcover' , 'windspeed','sunset','moonphase', 'description']
    
    weather_df = weather_df.drop(columns=drop_columns, errors='ignore')
    weather_df = weather_df.rename(columns={
        'tempmin': 'station_min_temp_c',
        'temp': 'station_avg_temp_c',
        'precip': 'station_precip_mm',
        'dew': 'reanalysis_dew_point_temp_k',
        'humidity': 'reanalysis_specific_humidity_g_per_kg',
        'tempmax': 'station_max_temp_c',
        'week_of_year': 'weekofyear'
    })
    
    celsius_to_kelvin_cols = ["reanalysis_dew_point_temp_k"]
    for col in celsius_to_kelvin_cols:
        weather_df[col] += 273.15
    
    return weather_df, None

def get_predictions(weather_df, model):
    features = weather_df[['year','weekofyear' ,'reanalysis_dew_point_temp_k' , 'reanalysis_specific_humidity_g_per_kg',
                        'station_avg_temp_c', 'station_max_temp_c' , 'station_min_temp_c' , 'station_precip_mm']]
    dmatrix = xgb.DMatrix(features)

    # Real-time predictions
    real_time_predictions = model.predict(dmatrix)
    real_time_predictions = [int(round(x)) for x in real_time_predictions]

    # Future predictions
    last_row = weather_df.iloc[-1][['year', 'weekofyear', 'reanalysis_dew_point_temp_k',
                                    'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',
                                    'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']]
    last_row = pd.DataFrame([last_row.values], columns=last_row.index)
    dtest = xgb.DMatrix(last_row)
    future_predictions = [int(round(model.predict(dtest)[0])) for _ in range(10)]

    return real_time_predictions, future_predictions

@app.route('/predict_IQ', methods=['POST'])
def predict_IQ():
    weather_df, error = process_weather_data("iquitos")
    if error:
        return jsonify(error), 500
    
    real_time_predictions, future_predictions = get_predictions(weather_df, model_IQ)
    
    predictions = {
        "city": "iquitos",
        "real_time": real_time_predictions,
        "future_horizon": future_predictions
    }
    
    return jsonify(predictions)

@app.route('/predict_SJ', methods=['POST'])
def predict_SJ():
    weather_df, error = process_weather_data("sanjuan")
    if error:
        return jsonify(error), 500
    
    real_time_predictions, future_predictions = get_predictions(weather_df, model_SJ)
    
    predictions = {
        "city": "sanjuan",
        "real_time": real_time_predictions,
        "future_horizon": future_predictions
    }
    
    return jsonify(predictions)

@app.route('/predict_both', methods=['POST'])
def predict_both():
    # Process Iquitos data
    iq_weather_df, error = process_weather_data("iquitos")
    if error:
        return jsonify(error), 500
    
    iq_real_time, iq_future = get_predictions(iq_weather_df, model_IQ)
    
    # Process San Juan data
    sj_weather_df, error = process_weather_data("sanjuan")
    if error:
        return jsonify(error), 500
    
    sj_real_time, sj_future = get_predictions(sj_weather_df, model_SJ)
    
    all_predictions = {
        "iquitos": {
            "real_time": iq_real_time,
            "future_horizon": iq_future
        },
        "sanjuan": {
            "real_time": sj_real_time,
            "future_horizon": sj_future
        }
    }
    
    return jsonify(all_predictions)

if __name__ == '__main__':
    app.run(debug=True)