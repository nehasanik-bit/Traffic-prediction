"""Train a Random Forest on traffic_data.csv and save it to model/."""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = ["hour", "day_of_week", "is_holiday", "weather", "vehicle_count", "avg_speed"]
TARGET = "traffic_condition"
LABELS = ["Low", "Medium", "Heavy"]

df = pd.read_csv("traffic_data.csv").dropna()

# One-hot encode the weather column; remember the final column order
X = pd.get_dummies(df[FEATURES], columns=["weather"])
y = df[TARGET]
columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}\n")
print(classification_report(y_test, pred, labels=LABELS))
print("Confusion matrix (rows=actual, cols=predicted; order Low, Medium, Heavy):")
print(confusion_matrix(y_test, pred, labels=LABELS))

print("\nFeature importance:")
for name, imp in sorted(zip(columns, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:<20}{imp:.3f}")

os.makedirs("model", exist_ok=True)
joblib.dump({"model": model, "columns": columns}, "model/traffic_model.joblib")
print("\nSaved model/traffic_model.joblib")
