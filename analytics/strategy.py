import numpy as np
from scipy.signal import argrelextrema


def generate_trading_strategy(forecast):
    """
    Анализ временного ряда:
    - локальные минимумы → BUY
    - локальные максимумы → SELL
    Возвращает список словарей вида {'action': 'BUY'/'SELL', 'price': float, 'index': int}
    """
    forecast = np.array(forecast)

    # локальные минимумы и максимумы
    minima_idx = argrelextrema(forecast, np.less)[0]
    maxima_idx = argrelextrema(forecast, np.greater)[0]

    strategy = []
    for i in minima_idx:
        strategy.append({'action': 'BUY', 'price': forecast[i], 'index': i})
    for i in maxima_idx:
        strategy.append({'action': 'SELL', 'price': forecast[i], 'index': i})

    strategy = sorted(strategy, key=lambda x: x['index'])
    return strategy
