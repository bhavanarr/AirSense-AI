
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import create_features


def train_xgboost(df):
    df = create_features(df, horizon=1)

    weather_features = [
        "temperature", "humidity", "wind_speed",
        "pressure", "precipitation"
    ]

    weather_features = [
        col for col in weather_features if col in df.columns
    ]

    lag_features = [
        f"pm25_lag_{lag}" for lag in [1, 2, 3, 6, 12, 24]
    ]

    features = lag_features + [
        "hour", "day_of_week", "month"
    ] + weather_features

    df = df.dropna(subset=features + ["target_pm25"])
    df = df.sort_values("timestamp")

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train[features]
    y_train = train["target_pm25"]

    X_test = test[features]
    y_test = test["target_pm25"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/xgboost.pkl")

    results = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "city": test["city"].values,
        "station_id": test["station_id"].values,
        "actual_pm25": y_test.values,
        "predicted_pm25": predictions
    })

    return model, metrics, results
