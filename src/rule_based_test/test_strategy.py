import matplotlib.pyplot as plt
import pandas as pd

from src.backtester import run_backtest
from src.config import load_settings
from src.data_loader.data_loader import load_prepared
from src.feature_engineer import add_indicators
from src.rule_based_test.strategy import generate_signals

if __name__ == "__main__":
    settings = load_settings()
    df = add_indicators(load_prepared("AAPL", settings), settings)
    df = generate_signals(df)
    result = run_backtest(df, settings, symbol="AAPL")
    df = df.copy()
    df["equity"] = result.equity

    print(df[["close", "signal"]].tail())
    print("\nFinal Equity:", result.final_equity)
    print("Total Trades:", len(result.trades))
    wins = sum(1 for t in result.trades if t.pnl > 0)
    print("Win Rate:", (wins / len(result.trades)) if result.trades else 0.0)

    plt.plot(result.equity)
    plt.title("EMA baseline equity")
    plt.show()
