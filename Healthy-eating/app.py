from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("healthy_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict_page():
    return render_template("predict.html")

@app.route('/result', methods=["POST"])
def result():
    if request.method == "POST":
        try:
            input_data = request.form.to_dict()
            numeric_cols = ['calories','protein_g','carbs_g','fat_g','fiber_g',
                            'sugar_g','sodium_mg','cholesterol_mg','serving_size_g',
                            'prep_time_min','cook_time_min','rating']
            for col in numeric_cols:
                input_data[col] = float(input_data[col])
            df = pd.DataFrame([input_data])
            df = pd.get_dummies(df)
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0 
            df = df[feature_columns]  
            features_scaled = scaler.transform(df)
            prediction = model.predict(features_scaled)[0]
            output = "Healthy Choice" if prediction == 1 else "Not Healthy Choice"

            return render_template("result.html", prediction_text=output)

        except Exception as e:
            return render_template("result.html", prediction_text=f"Error: {str(e)}")
if __name__ == "__main__":
    app.run(debug=True)
