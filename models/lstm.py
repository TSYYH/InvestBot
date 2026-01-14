import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .base import BaseModel


class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class LSTMModel(BaseModel):
    """
    LSTM модель для прогнозирования временных рядов
    """
    name = "LSTM"
    model_type = "dl"

    def __init__(
        self,
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str | None = None
    ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        self.model = LSTMNet(
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """
        Обучение модели
        """
        series = data["close"].values.reshape(-1, 1)

        if len(series) <= self.seq_len:
            raise ValueError("Time series too short for given seq_len")

        # scaling
        series_scaled = self.scaler.fit_transform(series).flatten()

        split_idx = int(len(series_scaled) * 0.8)
        train_series = series_scaled[:split_idx]
        test_series = series_scaled[split_idx - self.seq_len:]

        train_ds = TimeSeriesDataset(train_series, self.seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for x, y in train_loader:
                x = x.unsqueeze(-1).to(self.device)
                y = y.to(self.device)

                preds = self.model(x)
                loss = self.criterion(preds, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

        self.test_series = test_series
        self.is_fitted = True

    def evaluate(self) -> float:
        """
        RMSE на тестовом отрезке (1-step ahead)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        self.model.eval()
        preds = []
        true = []

        with torch.no_grad():
            for i in range(len(self.test_series) - self.seq_len):
                x = self.test_series[i: i + self.seq_len]
                y_true = self.test_series[i + self.seq_len]

                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                y_pred = self.model(x_t).cpu().item()

                preds.append(y_pred)
                true.append(y_true)

        preds = self.scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

        true = self.scaler.inverse_transform(
            np.array(true).reshape(-1, 1)
        ).flatten()

        rmse = np.sqrt(mean_squared_error(true, preds))
        return rmse

    def predict(self, last_values: np.ndarray, steps: int = 30) -> np.ndarray:
        """
        Итеративный прогноз на steps шагов вперед

        :param last_values: последние seq_len значений ряда (в исходном масштабе)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        history = self.scaler.transform(
            last_values.reshape(-1, 1)
        ).flatten().tolist()

        forecast = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                x = torch.tensor(
                    history[-self.seq_len:],
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1).to(self.device)

                y_pred = self.model(x).cpu().item()
                history.append(y_pred)
                forecast.append(y_pred)

        forecast = self.scaler.inverse_transform(
            np.array(forecast).reshape(-1, 1)
        ).flatten()

        return forecast
