def calculate_variance(returns):
    squared_log_price_return = returns ** 2
    var = squared_log_price_return.rolling(window=5).sum() / 5.0

    return var.dropna()