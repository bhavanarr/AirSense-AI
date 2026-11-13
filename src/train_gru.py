
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from torch.utils.data import TensorDataset, DataLoader


class AirSenseGRU(nn.Module):

    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


def train_gru(
    df,
    lookback=24,
    epochs=30,
    batch_size=32,
    cutoff=None
):

    df = df.copy()

    required_columns = ["timestamp", "station_id", "pm25"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    df = df.dropna(
        subset=["timestamp", "station_id", "pm25"]
    )

    df = df.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    # Use the same chronological cutoff as the tree models
    if cutoff is None:
        cutoff = df["timestamp"].quantile(0.8)

    cutoff = pd.Timestamp(cutoff)

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    feature_names = [
        "pm25", "hour", "day_of_week", "month"
    ]

    X_train, y_train = [], []
    X_test, y_test = [], []

    test_timestamps = []
    test_stations = []
    test_cities = []
    city_lookup = {}

    # Build sequences separately for each station
    for station, group in df.groupby("station_id"):

        group = group.sort_values("timestamp")

        series = group.set_index("timestamp")["pm25"]

        city = (
            group["city"].iloc[0]
            if "city" in group.columns
            else "Unknown"
        )

        city_lookup[station] = city

        # Map readings by their exact timestamps
        for origin_time, current_value in series.items():

            target_time = origin_time + pd.Timedelta(hours=1)

            # Target must exist exactly one hour later
            if target_time not in series.index:
                continue

            # Build the exact 24-hour input window
            window_times = pd.date_range(
                end=origin_time,
                periods=lookback,
                freq="h"
            )

            window_values = series.reindex(window_times)

            # Skip windows containing missing hourly readings
            if window_values.isna().any():
                continue

            sequence = pd.DataFrame({
                "pm25": window_values.to_numpy(),
                "hour": window_times.hour,
                "day_of_week": window_times.day_of_week,
                "month": window_times.month
            })

            target = series.loc[target_time]

            if origin_time <= cutoff:

                X_train.append(
                    sequence[feature_names].to_numpy()
                )

                y_train.append(target)

            else:

                X_test.append(
                    sequence[feature_names].to_numpy()
                )

                y_test.append(target)

                test_timestamps.append(target_time)
                test_stations.append(station)
                test_cities.append(city)

    if not X_train or not X_test:
        raise ValueError(
            "Not enough complete hourly sequences for training "
            "and testing. Try a smaller lookback."
        )

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    # Fit input scaler using training sequences only
    scaler = StandardScaler()

    scaler.fit(
        X_train.reshape(-1, len(feature_names))
    )

    X_train_scaled = scaler.transform(
        X_train.reshape(-1, len(feature_names))
    ).reshape(X_train.shape)

    X_test_scaled = scaler.transform(
        X_test.reshape(-1, len(feature_names))
    ).reshape(X_test.shape)

    # Fit target scaler using training targets only
    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.reshape(-1, 1)
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AirSenseGRU(
        input_size=len(feature_names),
        hidden_size=64,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    # Train GRU
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    # Generate test predictions
    model.eval()

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        scaled_predictions = (
            model(X_test_tensor).cpu().numpy()
        )

    predictions = target_scaler.inverse_transform(
        scaled_predictions
    ).flatten()

    # Evaluate model
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions)
    }

    results = pd.DataFrame({
        "timestamp": test_timestamps,
        "city": test_cities,
        "station_id": test_stations,
        "actual_pm25": y_test,
        "predicted_pm25": predictions
    })

    # Save model and scalers
    project_root = Path(__file__).resolve().parents[1]

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        models_dir / "gru_model.pth"
    )

    joblib.dump(
        scaler,
        models_dir / "gru_scaler.pkl"
    )

    joblib.dump(
        target_scaler,
        models_dir / "gru_target_scaler.pkl"
    )

    print("\nGRU model and scalers saved successfully!")
    print("Training sequences:", len(X_train))
    print("Testing sequences:", len(X_test))
    print("Forecast horizon: 1 hour")
    print("Input features:", feature_names)

    return model, metrics, results
