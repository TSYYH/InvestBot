import numpy as np


def make_forecast(model, data, days = 30):
    series = data["close"]

    if model.model_type == "ml_lag":
        last_values = series.values[-model.n_lags:]
        return model.predict(last_values, steps=days)

    elif model.model_type == "stat":
        return model.predict(steps=days)

    elif model.model_type == "dl":
        last_values = series.values[-model.seq_len:]
        return model.predict(last_values, steps=days)

    else:
        raise ValueError(f"Unknown model type: {model.model_type}")

