# data/data_fetcher.py
import ccxt
import pandas as pd
import sqlite3
import os
from datetime import datetime
from utils.indicators import calculate_indicators  # Your existing file


class DataFetcher:
    def __init__(self, db_path: str = "crypto_data.db"):
        self.db_path = db_path
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},  # or 'spot' if you prefer
            }
        )
        self._init_db()

    def _get_table_name(self, symbol: str, timeframe: str) -> str:
        """Generate table name from symbol and timeframe"""
        # Remove special characters and replace / with _
        clean_symbol = symbol.replace("/", "_").replace("-", "_")
        return f"ohlcv_{clean_symbol}_{timeframe}"

    def _init_db(self):
        """Create DB - tables will be created dynamically as needed"""
        # Just ensure the database file exists
        conn = sqlite3.connect(self.db_path)
        conn.close()

    def _create_table_if_not_exists(self, symbol: str, timeframe: str):
        """Create table for specific symbol and timeframe if it doesn't exist"""
        table_name = self._get_table_name(symbol, timeframe)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp INTEGER PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                sma_20 REAL,
                ema_12 REAL,
                ema_26 REAL
            )
        """
        )
        conn.commit()
        conn.close()

    def _get_last_timestamp(self, symbol: str, timeframe: str):
        """Get last timestamp for specific symbol and timeframe"""
        table_name = self._get_table_name(symbol, timeframe)
        conn = sqlite3.connect(self.db_path)

        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
        """
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            conn.close()
            return None

        query = f"SELECT MAX(timestamp) FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.iloc[0, 0] if not df.empty and pd.notna(df.iloc[0, 0]) else None

    def _save_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to specific symbol-timeframe table"""
        if df.empty:
            return

        # Create table if it doesn't exist
        self._create_table_if_not_exists(symbol, timeframe)

        df = df.copy()
        table_name = self._get_table_name(symbol, timeframe)

        # Ensure we have all required indicator columns
        required_columns = [
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "sma_20",
            "ema_12",
            "ema_26",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Add missing columns with None values

        # Select and order columns for database
        db_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ] + required_columns
        df = df[db_columns]

        conn = sqlite3.connect(self.db_path)

        # Use INSERT OR REPLACE to handle updates
        placeholders = ", ".join(["?"] * len(db_columns))
        columns_str = ", ".join(db_columns)

        for _, row in df.iterrows():
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """,
                tuple(row[col] for col in db_columns),
            )

        conn.commit()
        conn.close()

    def fetch_and_store(
        self, symbol: str, timeframe: str, since: int = None, limit: int = 1000
    ):
        """Fetch new candles and store them in specific table"""
        binance_symbol = symbol.replace("/", "")  # BTCUSDT
        last_ts = self._get_last_timestamp(symbol, timeframe)

        # If we have data, start from last candle + 1ms
        if last_ts:
            since = last_ts + 1
            print(
                f"Updating {symbol} {timeframe} from {datetime.utcfromtimestamp(last_ts/1000)}"
            )

        all_data = []
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    binance_symbol, timeframe, since=since, limit=limit
                )
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                print(f"   Fetched {len(ohlcv)} candles → Total: {len(all_data)}")
                if len(ohlcv) < limit:
                    break
            except Exception as e:
                print(f"Error fetching {symbol} {timeframe}: {e}")
                break

        if all_data:
            df = pd.DataFrame(
                all_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = calculate_indicators(df)  # Adds RSI, MACD, BBands, etc.
            df.reset_index(inplace=True)
            df["timestamp"] = df["timestamp"].astype("int64") // 10**6  # to ms int
            self._save_to_db(df, symbol, timeframe)
            print(f"Stored {len(df)} candles for {symbol} {timeframe}")

    def get_stored_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from specific symbol-timeframe table"""
        table_name = self._get_table_name(symbol, timeframe)
        conn = sqlite3.connect(self.db_path)

        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
        """
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            print(f"No table found for {symbol} {timeframe}")
            conn.close()
            return pd.DataFrame()

        query = f"SELECT * FROM {table_name} ORDER BY timestamp"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.name = f"{symbol}_{timeframe}"
        return df

    def list_available_tables(self):
        """List all available tables in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'ohlcv_%'
        """
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def fetch_multiple_symbols_data(
        self, symbols, timeframes=None, start_date="2025-01-01T00:00:00Z"
    ):
        """Main function – download/update multiple symbols & timeframes"""
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h", "4h"]  # Default for ML training

        since = int(pd.Timestamp(start_date).timestamp() * 1000)

        for symbol in symbols:
            for tf in timeframes:
                print(f"\nFetching {symbol} - {tf}")
                self.fetch_and_store(symbol, tf, since=since)

    def close(self):
        pass  # For compatibility


# Quick test (run this file directly)
if __name__ == "__main__":
    fetcher = DataFetcher()
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XAUUSDT"]
    fetcher.fetch_multiple_symbols_data(
        symbols, timeframes=["5m", "15m", "30m", "1h", "4h"]
    )
    print("\nFirst download complete! crypto_data.db is ready for ML training")

    # Show available tables
    tables = fetcher.list_available_tables()
    print("Available tables:", tables)

    # Example: Load BTC 1h data
    df = fetcher.get_stored_data("BTC/USDT", "1h")
    print(f"Loaded BTC/USDT 1h data: {len(df)} rows")
