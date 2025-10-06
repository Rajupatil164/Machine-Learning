from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and encoders
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        gender = request.form['gender']
        married = request.form['married']
        education = request.form['education']
        employment = request.form['employment']
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_history = request.form['credit_history']

        # Encode categorical values
        gender = label_encoders['Gender'].transform([gender])[0]
        married = label_encoders['Married'].transform([married])[0]
        education = label_encoders['Education'].transform([education])[0]
        employment = label_encoders['Employment'].transform([employment])[0]
        credit_history = label_encoders['Credit_History'].transform([credit_history])[0]

        # Prepare input for prediction
        input_data = np.array([[gender, married, education, employment, income, loan_amount, credit_history]])

        # Predict
        prediction = model.predict(input_data)[0]
        result = "Eligible for Loan" if prediction == 1 else "Not Eligible for Loan"

        return render_template('result.html', result=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
