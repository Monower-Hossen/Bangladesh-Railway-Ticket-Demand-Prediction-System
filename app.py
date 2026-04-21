from flask import Flask, request, render_template
import pandas as pd
import os
import holidays
from datetime import datetime, timedelta
from src.ml_projects.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.ml_projects.config.configuration import ConfigurationManager
from src.ml_projects.constants import APP_HOST, APP_PORT

app = Flask(__name__)

# Initialize configuration
config_manager = ConfigurationManager()
ingestion_config = config_manager.get_data_ingestion_config()
bd_holidays = holidays.Bangladesh()
_df_feature_store = None

def get_feature_store():
    global _df_feature_store
    if _df_feature_store is None:
        path = ingestion_config.feature_store_file_path
        if os.path.exists(path):
            _df_feature_store = pd.read_csv(path)
    return _df_feature_store

def get_unique_values(column_name: str) -> list:
    try:
        df = get_feature_store()
        if df is not None and column_name in df.columns:
            return sorted([str(x) for x in df[column_name].dropna().unique()])
    except Exception as e:
        print(f"Error fetching {column_name}: {e}")
    return []

def get_all_stations():
    return sorted(list(set(get_unique_values('From') + get_unique_values('To'))))

def get_page_context():
    return {
        "stations": get_all_stations(),
        "trains": get_unique_values('Train_Name'),
        "coaches": get_unique_values('Coach'),
        "payment_methods": get_unique_values('Payment_Method')
    }

def analyze_demand_insights(prediction, departure_time, current_fare, from_st, to_st, group_size=1, journey_date=None, issue_date=None, forecast_data=None):
    """Generates demand insights with robust error handling."""
    insights = {}
    capacity = 80
    
    # 1. Booking Pattern & Anomaly
    try:
        d1 = datetime.strptime(journey_date, '%Y-%m-%d')
        d2 = datetime.strptime(issue_date, '%Y-%m-%d')
        lead_time = (d1 - d2).days
        insights['booking_pattern'] = f"Booked {lead_time} days in advance."
        insights['is_anomaly'] = bool(prediction > (capacity * 1.5))
    except:
        insights['booking_pattern'] = "N/A"
        insights['is_anomaly'] = False

    # 2. Peak Hour
    try:
        hour = int(departure_time.split(':')[0])
        is_peak = (7 <= hour <= 10) or (17 <= hour <= 21)
        insights['peak_status'] = "Peak Hour" if is_peak else "Off-Peak"
    except:
        insights['peak_status'] = "N/A"

    # 3. Demand Forecasting
    demand_pct = (prediction / capacity) * 100
    insights['demand_level'] = "Critical" if demand_pct > 90 else ("Medium" if demand_pct > 60 else "Low")
    insights['allocation'] = "Deploy Extra Coach" if demand_pct > 90 else "Standard Allocation"
    
    # 4. Fare Strategy
    total_fare = current_fare * group_size
    if demand_pct > 85:
        insights['fare_strategy'] = f"Surge (+15%): BDT {round(current_fare * 1.15, 2)}"
    elif demand_pct < 30:
        insights['fare_strategy'] = f"Promo (-10%): BDT {round(current_fare * 0.9, 2)}"
    else:
        insights['fare_strategy'] = f"Standard Fare: BDT {current_fare}"

    # 5. Chart Data (Safely generated)
    try:
        if forecast_data:
            insights['chart_data'] = forecast_data
        else:
            base_date = datetime.strptime(journey_date, '%Y-%m-%d')
            labels = [(base_date + timedelta(days=i)).strftime('%b %d') for i in range(-3, 4)]
            values = [round(prediction * (0.8 + (i*0.05)), 1) for i in range(7)]
            insights['chart_data'] = {"labels": labels, "values": values}
    except:
        insights['chart_data'] = {"labels": [], "values": []}

    return insights

@app.route('/')
def index():
    return render_template('index.html', results=None, form_data={}, **get_page_context())

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        # 1. Capture Form Inputs
        train_name = request.form.get('Train_Name')
        from_st = request.form.get('From')
        to_st = request.form.get('To')
        coach = request.form.get('Coach')
        fare = float(request.form.get('Fare'))
        journey_date = request.form.get('Journey_Date')
        departure_time = request.form.get('Departure_Time')
        payment_method = request.form.get('Payment_Method', 'Online')

        # 2. Prepare Multi-Day Demand Forecast (7-day window starting from selected journey date)
        base_date_obj = datetime.strptime(journey_date, '%Y-%m-%d')
        forecast_dfs = []
        current_date_str = datetime.now().strftime('%Y-%m-%d')
        current_time_str = datetime.now().strftime('%H:%M:%S')

        for i in range(7):
            target_date = (base_date_obj + timedelta(days=i)).strftime('%Y-%m-%d')
            day_data = CustomData(
                Train_Name=train_name,
                From=from_st,
                To=to_st,
                Coach=coach,
                Fare=fare,
                Journey_Date=target_date,
                Departure_Time=departure_time,
                Issue_Date=current_date_str,
                Issue_Time=current_time_str,
                Group_Size=1,
                Search_Volume=100,
                Is_Holiday=1 if datetime.strptime(target_date, '%Y-%m-%d') in bd_holidays else 0,
                Payment_Method=payment_method
            )
            forecast_dfs.append(day_data.get_data_as_data_frame())

        # Combine all days into one DataFrame for batch prediction
        full_forecast_df = pd.concat(forecast_dfs, ignore_index=True)

        # 3. Get Batch Predictions and Insights
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(full_forecast_df)
        
        # The primary result is for the specific selected date (index 0)
        prediction_val = float(predictions[0])
        
        # Construct actual forecast data for the chart
        forecast_results = {
            "labels": [(base_date_obj + timedelta(days=i)).strftime('%b %d') for i in range(7)],
            "values": [round(float(p), 1) for p in predictions]
        }
        
        insights = analyze_demand_insights(
            prediction=prediction_val, 
            departure_time=departure_time, 
            current_fare=fare, 
            from_st=from_st, 
            to_st=to_st,
            journey_date=journey_date,
            issue_date=current_date_str,
            forecast_data=forecast_results
        )

        return render_template(
            'index.html', 
            results=prediction_val, 
            insights=insights, 
            form_data=request.form.to_dict(), # Method executed: returns dict
            **get_page_context()
        )

    except Exception as e:
        return render_template(
            'index.html', 
            error=str(e), 
            results=None, 
            insights=None,
            form_data=request.form.to_dict(), 
            **get_page_context()
        )

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)