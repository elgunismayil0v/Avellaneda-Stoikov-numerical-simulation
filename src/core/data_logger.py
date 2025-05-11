# core/data_logger.py
import pandas as pd
class DataLogger:
    def __init__(self):
        self.data = {
            'mid_prices': [],
            'bid_prices': [],
            'ask_prices': [],
            'reservation_prices': [],
            'inventory': [],
            'cash': []
        }

    def log(self, key: str, value):
        self.data[key].append(value)

    def get_dataframe(self):
        return pd.DataFrame(self.data)
