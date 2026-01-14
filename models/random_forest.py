import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .base import BaseModel


class RandomForestLagModel(BaseModel):
    """
    Random Forest для временных рядов с лаговыми признаками
    """
    name = "RandomForest (lags)"
    model_type = "ml_lag"

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = 42
    ):
        self.n_lags = n_lags
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def _make_lag_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Создание DataFrame с лаговыми признаками
        """
        df = pd.DataFrame({"y": series})

        for lag in range(1, self.n_lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)

        df.dropna(inplace=True)
        return df

    def fit(self, data: pd.DataFrame):
        """
        Обучение модели
        """
        series = data["close"]
        df = self._make_lag_features(series)

        X = df.drop(columns="y")
        y = df["y"]

        # train/test split по времени (80/20)
        split_idx = int(len(df) * 0.8)
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> float:
        """
        RMSE на тестовом множестве
        """
        preds = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        return rmse

    def predict(self, last_values: np.ndarray, steps: int = 30) -> np.ndarray:
        """
        Итеративный прогноз на steps дней вперёд

        :param last_values: последние n_lags значений ряда
        :param steps: горизонт прогноза
        """
        history = list(last_values)
        forecast = []

        for _ in range(steps):
            x = np.array(history[-self.n_lags:]).reshape(1, -1)

            # превращаем в DataFrame с именами колонок
            x_df = pd.DataFrame(x, columns=[f"lag_{i + 1}" for i in range(self.n_lags)])

            y_pred = self.model.predict(x_df)[0]

            forecast.append(y_pred)
            history.append(y_pred)

        return np.array(forecast)
