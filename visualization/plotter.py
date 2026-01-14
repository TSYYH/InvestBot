import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_forecast(history: pd.DataFrame, forecast, save_dir: str = "plots") -> str:
    """
    Строит график: факт + прогноз и сохраняет в файл

    :param history: DataFrame с колонками ['date', 'close']
    :param forecast: np.ndarray с прогнозом
    :param save_dir: директория для сохранения графика
    :return: путь к сохранённому файлу
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])

    last_date = history["date"].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(forecast),
        freq="B"  # business days
    )

    plt.figure(figsize=(12, 6))

    # Исторические данные
    plt.plot(history["date"], history["close"], label="Исторические данные")

    # Прогноз
    plt.plot(forecast_dates, forecast, label="Прогноз", linestyle="--")

    # Граница прогноза
    plt.axvline(x=last_date, linestyle=":", label="Начало прогноза")

    plt.title("Прогноз цены акции")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)

    file_path = Path(save_dir) / "forecast.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    return str(file_path)
