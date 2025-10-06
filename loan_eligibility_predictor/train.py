import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
# Example dataset path: 'loan_data.csv'
# Columns: Gender, Married, Education, Employment, ApplicantIncome, LoanAmount, Credit_History, Loan_Status
data = pd.read_csv('loan_data.csv')

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Education', 'Employment', 'Credit_History']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Target variable
X = data.drop('Loan_Status', axis=1)
y = LabelEncoder().fit_transform(data['Loan_Status'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model training complete. Files saved: loan_model.pkl, label_encoders.pkl")
