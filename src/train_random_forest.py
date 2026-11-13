
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from feature_engineering import create_features


def train_random_forest(df):

    df = df.copy()

    # Validate required columns
    required = [
        "timestamp", "city", "station_id", "pm25"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create lag and time-based features
    df = create_features(df, horizon=1)

    weather_features = [
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "precipitation"
    ]

    weather_features = [
        col for col in weather_features if col in df.columns
    ]

    lag_features = [
        f"pm25_lag_{lag}" for lag in [1, 2, 3, 6, 12, 24]
    ]

    features = (
        lag_features
        + ["hour", "day_of_week", "month"]
        + weather_features
    )

    # Remove rows with missing model features
    df = df.dropna(subset=features + ["target_pm25"])

    if df.empty:
        raise ValueError("No usable rows remain after feature creation.")

    # Use one chronological cutoff for all stations
    cutoff = df["timestamp"].quantile(0.8)

    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]

    if train.empty or test.empty:
        raise ValueError("Not enough data for chronological splitting.")

    X_train = train[features]
    y_train = train["target_pm25"]

    X_test = test[features]
    y_test = test["target_pm25"]

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Evaluate model
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    }

    # Save trained model
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        model,
        models_dir / "random_forest.pkl"
    )

    # Save prediction results
    results = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "city": test["city"].values,
        "station_id": test["station_id"].values,
        "actual_pm25": y_test.values,
        "predicted_pm25": predictions
    })

    print("Random Forest training completed!")
    print("Model saved successfully!")

    return model, metrics, results
