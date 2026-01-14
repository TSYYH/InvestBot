import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def load_stock_data(ticker: str, days: int = 365 * 2) -> pd.DataFrame:
    """
    Загружает исторические котировки акций за последние N лет.

    :param ticker: тикер акции (AAPL, MSFT и т.д.)
    :param days: глубина истории в днях
    :return: DataFrame с колонками:
             ['date', 'close']
    :raises ValueError: если данные не получены
    """

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    try:
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке данных: {e}")

    if data.empty:
        raise ValueError(f"Нет данных по тикеру {ticker}")

    # Оставляем только цену закрытия
    data = data.reset_index()[["Date", "Close"]]
    data.columns = ["date", "close"]

    return data
