from .random_forest import RandomForestLagModel
from .arima import SARIMAModel
from .lstm import LSTMModel


def select_best_model(data):
    """
    Обучает все модели и выбирает лучшую по метрике
    """
    models = [RandomForestLagModel(), SARIMAModel(), LSTMModel()]
    #models = [RandomForestLagModel()]
    #models = [ARIMAModel()]
    #models = [LSTMModel()]

    best_model = None
    best_metric = float("inf")

    for model in models:
        model.fit(data)
        metric = model.evaluate()

        if metric < best_metric:
            best_metric = metric
            best_model = model

    return best_model, best_metric
