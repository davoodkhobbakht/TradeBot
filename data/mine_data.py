# mine_data.py – Run: python3 mine_data.py
import pandas as pd
from data.data_fetcher import DataFetcher
import matplotlib.pyplot as plt

fetcher = DataFetcher()
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
timeframe = "1d"  # Start with daily – switch to '1h' for deeper

for symbol in symbols:
    df = fetcher.get_stored_data(symbol, timeframe)
    if df.empty:
        print(f"No data for {symbol} {timeframe}")
        continue

    print(f"\n🔍 Mining {symbol} ({timeframe}) – {len(df)} candles")
    print("📊 Basic Stats:")
    print(df[["close", "volume", "RSI", "MACD", "ADX"]].describe())

    # Correlations (mine for patterns)
    corr = df[["close", "RSI", "MACD", "ADX", "BB_Bandwidth"]].corr()
    print("\n📈 Correlations:")
    print(corr)

    # Plot: Price + RSI + Volume (save to file)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1.plot(df.index, df["close"], label="Close", color="blue")
    ax1.set_title(f"{symbol} Price")
    ax2.plot(df.index, df["RSI"], label="RSI", color="green")
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")
    ax3.bar(df.index, df["volume"], label="Volume", color="gray")
    plt.tight_layout()
    plt.savefig(f'{symbol.replace("/", "_")}_{timeframe}_plot.png')
    print(f"💾 Plot saved: {symbol.replace('/', '_')}_{timeframe}_plot.png")

print("\nMining complete! Check plots & stats above.")
