
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_actual, y_predicted, model_name):

    metrics = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_actual, y_predicted),
        "RMSE": np.sqrt(
            mean_squared_error(y_actual, y_predicted)
        ),
        "R2": r2_score(y_actual, y_predicted)
    }

    return metrics


def compare_models(model_results):

    comparison = pd.DataFrame(model_results)

    comparison = comparison.sort_values(
        by="RMSE",
        ascending=True
    )

    return comparison
