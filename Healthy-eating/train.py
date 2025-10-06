import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv("healthy_eating_dataset.csv")

X = data.drop('is_healthy', axis=1)
y = data['is_healthy']
# One-hot encode categorical variables
le = pd.get_dummies(X.select_dtypes(include=['object']), drop_first=True)
X = X.select_dtypes(exclude=['object']).join(le)
feature_columns = X.columns
pickle.dump(feature_columns, open("feature_columns.pkl", "wb"))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open("healthy_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model, scaler, and feature columns saved successfully!")
print(f"Classification Report:\n{classification_report(y_test, model.predict(X_test))}")