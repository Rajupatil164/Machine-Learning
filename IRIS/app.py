from flask import Flask, request, render_template
import numpy as np
import joblib
from sklearn.datasets import load_iris

iris = load_iris()
model = joblib.load("iris_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        flower_name = iris.target_names[prediction]

        return render_template('index.html', prediction_text=f'Predicted Flower: {flower_name}')
    except:
        return render_template('index.html', prediction_text=f'Invalid Input')
    
if __name__ == "__main__":
    app.run(debug=True)
