
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination=0.02):
    """
    Detect unusual PM2.5 readings separately for each monitoring station.
    """

    data = df.copy()

    required_columns = ["station_id", "pm25"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce")

    data["anomaly_label"] = 1
    data["anomaly_score"] = float("nan")

    station_models = {}

    for station_id, group in data.groupby("station_id"):
        valid_index = group.index[group["pm25"].notna()]

        if len(valid_index) < 10:
            continue

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )

        values = data.loc[valid_index, ["pm25"]]

        model.fit(values)

        data.loc[valid_index, "anomaly_label"] = model.predict(values)
        data.loc[valid_index, "anomaly_score"] = model.decision_function(values)

        station_models[station_id] = model

    data["anomaly"] = data["anomaly_label"] == -1

    return data, station_models
