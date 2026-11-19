"""Generate a realistic synthetic traffic dataset -> traffic_data.csv

Replace this file's output with a real CSV later if you like; just keep the
same column names (or edit FEATURES in train_model.py).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 20000

hour = rng.integers(0, 24, N)
day = rng.integers(0, 7, N)                      # 0 = Monday ... 6 = Sunday
holiday = (rng.random(N) < 0.05).astype(int)
weather = rng.choice(["Clear", "Cloudy", "Rain", "Fog"], N, p=[0.45, 0.25, 0.22, 0.08])

# Base demand: rush-hour peaks on weekdays, milder midday bump on weekends
weekday = (day < 5) & (holiday == 0)
rush = np.exp(-((hour - 8.5) ** 2) / 4) + np.exp(-((hour - 17.5) ** 2) / 5)
midday = np.exp(-((hour - 13) ** 2) / 18)
demand = np.where(weekday, 0.25 + 0.95 * rush + 0.2 * midday, 0.15 + 0.5 * midday)
demand += rng.normal(0, 0.08, N)

weather_penalty = pd.Series(weather).map({"Clear": 0, "Cloudy": 0.03, "Rain": 0.12, "Fog": 0.18}).values
score = np.clip(demand + weather_penalty, 0, None)

vehicle_count = (score * 900 + rng.normal(0, 60, N) + 40).clip(10, 1500).round().astype(int)
avg_speed = (75 - score * 48 + rng.normal(0, 6, N)).clip(5, 90).round(1)

# Label from the combined score with noise so the task isn't trivially perfect
label_score = score + rng.normal(0, 0.07, N)
traffic = np.where(label_score < 0.45, "Low", np.where(label_score < 0.85, "Medium", "Heavy"))

df = pd.DataFrame({
    "hour": hour, "day_of_week": day, "is_holiday": holiday, "weather": weather,
    "vehicle_count": vehicle_count, "avg_speed": avg_speed, "traffic_condition": traffic,
})
df.to_csv("traffic_data.csv", index=False)
print(df.head(), "\n\nShape:", df.shape)
print("\nClass balance:\n", df["traffic_condition"].value_counts())
