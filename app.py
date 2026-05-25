from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model, scaler, and features names from your Colab exports
model = joblib.load("energy_model.pkl")
scaler = joblib.load("scaler.pkl")
# Use the features list from your Colab to ensure column alignment
try:
    feature_names = joblib.load('features.pkl')
except:
    feature_names = ['Household_Size', 'Avg_Temperature_C', 'Day', 'DayOfWeek', 'Temp_Household_Size_Interaction', 'Has_AC_encoded']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get raw inputs from the HTML form
        household_size = float(request.form['Household_Size'])
        temperature = float(request.form['Avg_Temperature_C'])
        has_ac = 1 if 'Has_AC' in request.form else 0
        
        # 2. Process Date
        date_str = request.form.get('Date')
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day = date_obj.day
        day_of_week = date_obj.weekday()

        # 3. Feature Engineering (Match Colab Logic)
        temp_household = temperature * household_size

        # 4. Build DataFrame
        input_data = pd.DataFrame([[
            household_size,
            temperature,
            day,
            day_of_week,
            temp_household,
            has_ac
        ]], columns=feature_names)

        # 5. Scale and Predict
        input_scaled = scaler.transform(input_data)
        log_prediction = model.predict(input_scaled)
        
        # 6. Inverse Transform (Log to Actual) as per Colab
        actual_prediction = np.expm1(log_prediction)[0]

        return render_template(
            'index.html',
            prediction_text=f"Predicted Energy Consumption: {actual_prediction:.2f} kWh",
            form_data=request.form  # Sending form data back to keep values
        )

    except Exception as e:
        print("Backend Error:", str(e))
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            form_data=request.form
        )

if __name__ == "__main__":
    app.run(debug=True)