# app.py
from flask import Flask, request, render_template
import numpy as np
import joblib

# Load trained model
model = joblib.load('diabetes_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        blood_pressure = float(request.form['blood_pressure'])
        glucose_level = float(request.form['glucose_level'])

        # Create feature array
        features = np.array([[age, bmi, blood_pressure, glucose_level]])

        # Predict
        prediction = model.predict(features)[0]
        output = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

        return render_template('index.html',
                               prediction_text=f'Patient is likely: {output}')
    except:
        return render_template('index.html',
                               prediction_text="Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)
