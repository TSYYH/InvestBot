def calculate_profit(strategy, amount):
    """
    Рассчитывает прибыль по стратегии.
    amount — сумма на первую покупку
    """
    balance = 0.0
    holding = 0.0  # количество актива в руках

    for trade in strategy:
        price = trade['price']
        if trade['action'] == 'BUY' and balance == 0.0:
            # покупаем на всю сумму
            holding = amount / price
            balance = 0.0
        elif trade['action'] == 'SELL' and holding > 0.0:
            # продаем всё
            balance = holding * price
            holding = 0.0

    # если после всех сделок остался актив — оцениваем по последней цене
    if holding > 0.0:
        balance = holding * strategy[-1]['price']

    return balance