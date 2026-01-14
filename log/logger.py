from datetime import datetime
from config import LOG_FILE


def log_request(user_id, ticker, amount, model_name, metric_value, profit):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now()}, "
            f"user_id={user_id}, "
            f"ticker={ticker}, "
            f"amount={amount}, "
            f"model={model_name}, "
            f"metric={metric_value}, "
            f"profit={profit}\n"
        )
