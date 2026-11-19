import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

bundle = joblib.load("model/traffic_model.joblib")
model, COLUMNS = bundle["model"], bundle["columns"]

WEATHER = ["Clear", "Cloudy", "Rain", "Fog"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_input(data):
    """Validate raw form/JSON data. Returns (clean_dict, error_message)."""
    try:
        hour = int(data.get("hour"))
        day = int(data.get("day_of_week"))
        holiday = int(data.get("is_holiday", 0))
        vehicles = int(data.get("vehicle_count"))
        speed = float(data.get("avg_speed"))
    except (TypeError, ValueError):
        return None, "Please fill in every field with a valid number."
    weather = data.get("weather")

    if not 0 <= hour <= 23:
        return None, "Hour must be between 0 and 23."
    if not 0 <= day <= 6:
        return None, "Pick a valid day of the week."
    if holiday not in (0, 1):
        return None, "Holiday must be Yes or No."
    if weather not in WEATHER:
        return None, "Pick a valid weather condition."
    if not 0 <= vehicles <= 5000:
        return None, "Vehicle count must be between 0 and 5000."
    if not 0 <= speed <= 150:
        return None, "Average speed must be between 0 and 150 km/h."

    return {"hour": hour, "day_of_week": day, "is_holiday": holiday,
            "weather": weather, "vehicle_count": vehicles, "avg_speed": speed}, None


def predict(clean):
    X = pd.get_dummies(pd.DataFrame([clean]), columns=["weather"])
    X = X.reindex(columns=COLUMNS, fill_value=0)   # match training columns exactly
    label = model.predict(X)[0]
    probs = dict(zip(model.classes_, model.predict_proba(X)[0]))
    return label, {k: round(float(probs.get(k, 0)) * 100, 1) for k in ("Low", "Medium", "Heavy")}


@app.route("/")
def home():
    return render_template("index.html", days=DAYS, weather=WEATHER)


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or request.form
    clean, error = parse_input(data)
    if error:
        return jsonify({"error": error}), 400
    label, probs = predict(clean)
    return jsonify({"prediction": label, "probabilities": probs})


if __name__ == "__main__":
    app.run(debug=True)
