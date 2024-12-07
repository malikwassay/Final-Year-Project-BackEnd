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

# Define city configurations
CITY_CONFIGS = {
    'iquitos': {'model': model_IQ, 'threshold': 12},
    'sanjuan': {'model': model_SJ, 'threshold': 20},
    'lima': {'model': model_SJ, 'threshold': 20},
    'cajamarca': {'model': model_SJ, 'threshold': 20},
    'pucallpa': {'model': model_SJ, 'threshold': 20},
    'tarapoto': {'model': model_SJ, 'threshold': 20}
}

app = Flask(__name__)
CORS(app)

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

    real_time_predictions = model.predict(dmatrix)
    real_time_predictions = [int(round(x)) for x in real_time_predictions]

    last_row = weather_df.iloc[-1][['year', 'weekofyear', 'reanalysis_dew_point_temp_k',
                                    'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',
                                    'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']]
    last_row = pd.DataFrame([last_row.values], columns=last_row.index)
    dtest = xgb.DMatrix(last_row)
    future_predictions = [int(round(model.predict(dtest)[0])) for _ in range(10)]

    return real_time_predictions, future_predictions

@app.route('/predict_city/<city>', methods=['POST'])
def predict_city(city):
    if city not in CITY_CONFIGS:
        return jsonify({"error": f"City {city} not supported"}), 400
    
    weather_df, error = process_weather_data(city)
    if error:
        return jsonify(error), 500
    
    real_time_predictions, future_predictions = get_predictions(weather_df, CITY_CONFIGS[city]['model'])
    
    predictions = {
        "city": city,
        "real_time": real_time_predictions,
        "future_horizon": future_predictions,
        "threshold": CITY_CONFIGS[city]['threshold']
    }
    
    return jsonify(predictions)

@app.route('/predict_all', methods=['POST'])
def predict_all():
    all_predictions = {}
    
    for city in CITY_CONFIGS:
        weather_df, error = process_weather_data(city)
        if error:
            all_predictions[city] = {"error": error}
            continue
        
        real_time_predictions, future_predictions = get_predictions(weather_df, CITY_CONFIGS[city]['model'])
        
        all_predictions[city] = {
            "real_time": real_time_predictions,
            "future_horizon": future_predictions,
            "threshold": CITY_CONFIGS[city]['threshold']
        }
    
    return jsonify(all_predictions)

if __name__ == '__main__':
    app.run(debug=True)