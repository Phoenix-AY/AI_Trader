import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd

load_dotenv("config/secrets.env")

api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),
    api_version='v2'
)

def fetch_data(symbol="AAPL"):
    bars = api.get_bars(
        symbol,
        TimeFrame.Minute,
        start="2025-11-01",
        end="2026-04-01"
    ).df

    filename = f"data/raw/{symbol}.csv"
    bars.to_csv(filename)

    print(f"Saved data to {filename}")
    print(bars.head())

if __name__ == "__main__":
    fetch_data()