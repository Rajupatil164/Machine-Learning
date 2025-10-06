from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

data = load_diabetes()
X = data.data
y = data.target
y = (y > y.mean()).astype(int)

# col 0 = age, col 2 = bmi, col 3 = bp, col 8 = s5 (proxy for glucose)
X_selected = X[:, [0, 2, 3, 8]]


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'diabetes_model.pkl')
print("Model trained on 4 features and saved as 'diabetes_model.pkl'")
