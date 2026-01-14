import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command

from config import BOT_TOKEN, HISTORY_DAYS
from data.data_loader import load_stock_data
from models.model_selector import select_best_model
from forecasting.forecast import make_forecast
from visualization.plotter import plot_forecast
from analytics.strategy import generate_trading_strategy
from analytics.profit import calculate_profit
from log.logger import log_request


async def start_handler(message: Message):
    await message.answer(
        "Привет! Я бот для прогнозирования акций и генерации простой торговой стратегии.\n\n"
        "Введите тикер и сумму инвестиции, например:\n"
        "AAPL 100000\n\n"
        "Бот рассчитает прогноз на 30 дней, предложит точки BUY/SELL и ориентировочную прибыль.\n\n"
        "⚠️ Важно: это не является инвестиционной рекомендацией. "
        "Вся информация предоставляется исключительно в образовательных целях."
    )


async def main_handler(message: Message):
    try:
        ticker, amount = message.text.split()
        amount = float(amount)
    except ValueError:
        await message.answer("Неверный формат. Используйте: TICKER AMOUNT")
        return

    await message.answer("Загружаю данные...")

    # Загрузка данных
    try:
        data = load_stock_data(ticker, HISTORY_DAYS)
    except ValueError as e:
        await message.answer(f"Нет данных по тикеру {ticker}")
        return

    # Обучение и выбор модели
    best_model, metric_value = select_best_model(data)

    # Прогнозирование
    forecast = make_forecast(best_model, data)

    # Рекомендации и прибыль
    strategy = generate_trading_strategy(forecast)
    profit = calculate_profit(strategy, amount)

    # Визуализация
    plot_path = plot_forecast(data, forecast)

    # Получаем текущую цену
    current_price = data["close"].values[-1]
    forecast_start = forecast[0]
    forecast_end = forecast[-1]

    # Изменение прогноза относительно текущего дня
    change_percent = (forecast_end - current_price) / current_price * 100
    change_sign = "↑" if change_percent > 0 else "↓"

    # Формируем текст стратегии
    strategy_text = []
    for trade in strategy:
        action = trade['action']
        price = trade['price']
        idx = trade['index']
        strategy_text.append(f"{idx + 1}: {action} @ {price:.2f}")

    strategy_str = "\n".join(strategy_text) if strategy_text else "Нет подходящей стратегии"

    # Формируем сообщение
    message_text = (
        f"📊 Выбранная модель: {best_model.name}\n"
        f"Метрика (RMSE): {metric_value:.2f}\n\n"
        f"Прогноз изменения цены за {len(forecast)} дней: {change_sign} {abs(change_percent):.2f}%\n\n"
        f"💡 Торговая стратегия:\n{strategy_str}\n\n"
        f"💰 Ориентировочная прибыль при инвестиции {amount:.2f}: {profit:.2f}"
    )

    await message.answer_photo(
        FSInputFile(plot_path),
        caption=message_text
    )

    # Логирование
    log_request(
        user_id=message.from_user.id,
        ticker=ticker,
        amount=amount,
        model_name=best_model.name,
        metric_value=metric_value,
        profit=profit
    )


async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    # Регистрация хендлеров
    dp.message.register(start_handler, Command("start"))
    dp.message.register(main_handler)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
