import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from .base import BaseModel


class SARIMAModel(BaseModel):
    """
    SARIMA модель
    """
    name = "SARIMA"
    model_type = "stat"

    def __init__(
        self,
        order=(5, 1, 0),
        seasonal_order=(0, 0, 0, 0)
    ):
        """
        :param order: (p, d, q)
        :param seasonal_order: (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order

        self.model = None
        self.fitted_model = None
        self.train_series = None
        self.test_series = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """
        Обучение SARIMA модели
        """
        series = data["close"]

        min_len = max(
            sum(self.order),
            sum(self.seasonal_order[:3]) * max(1, self.seasonal_order[3])
        )

        if len(series) <= min_len:
            raise ValueError("Time series too short for given SARIMA configuration")

        split_idx = int(len(series) * 0.8)
        self.train_series = series.iloc[:split_idx]
        self.test_series = series.iloc[split_idx:]

        self.model = SARIMAX(
            self.train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="c"
        )

        self.fitted_model = self.model.fit(disp=False)
        self.is_fitted = True

    def evaluate(self) -> float:
        """
        RMSE на тестовом отрезке
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        preds = self.fitted_model.forecast(steps=len(self.test_series))
        rmse = np.sqrt(mean_squared_error(self.test_series, preds))
        return rmse

    def predict(self, steps: int) -> np.ndarray:
        """
        Прогноз на steps дней вперед
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        forecast = self.fitted_model.forecast(steps=steps)
        return np.asarray(forecast)
