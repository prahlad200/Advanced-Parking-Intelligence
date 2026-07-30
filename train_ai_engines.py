import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import pickle

print("Generating 6 months of historical data...")
dates = pd.date_range(start="2026-01-01", end="2026-07-28", freq="h")
df = pd.DataFrame({"ds": dates})
df["hour"] = df["ds"].dt.hour
df["day_of_week"] = df["ds"].dt.dayofweek

# Simulate normal traffic
df["y"] = df["hour"].apply(lambda x: np.random.randint(50, 150) if 9 <= x <= 17 else np.random.randint(450, 543))

# Inject some fake "Anomalies" for the Isolation Forest to learn what is normal vs weird
# (e.g., parking lot completely full at 3 AM on a Sunday)
for i in range(20):
    random_idx = np.random.randint(0, len(df))
    df.loc[random_idx, "y"] = 543 

print("1. Training Facebook Prophet (Time-Series Forecaster)...")
prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
prophet_model.fit(df[["ds", "y"]])

print("2. Training Isolation Forest (Unsupervised Anomaly Detector)...")
iso_forest = IsolationForest(contamination=0.02, random_state=42)
X_iso = df[["hour", "day_of_week", "y"]]
iso_forest.fit(X_iso)

print("Saving Advanced AI Engines...")
with open("advanced_ai_models.pkl", "wb") as f:
    pickle.dump({"prophet": prophet_model, "isolation_forest": iso_forest}, f)

print("Success! 'advanced_ai_models.pkl' is ready.")