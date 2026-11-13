
import pandas as pd


def create_features(df, horizon=1):
    """
    Create time-based and PM2.5 lag features.
    horizon=1 predicts the next hourly PM2.5 value.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["city", "station_id", "timestamp"]
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    grouped = df.groupby(["city", "station_id"])["pm25"]

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}"] = grouped.shift(lag)

    df["target_pm25"] = grouped.shift(-horizon)

    return df
