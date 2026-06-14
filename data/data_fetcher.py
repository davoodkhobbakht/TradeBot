import ccxt
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import time
from utils.indicators import calculate_indicators

class DataFetcher:
    def __init__(self, db_path: str = "crypto_data.db", use_fallback: bool = True):
        self.db_path = db_path
        self.use_fallback = use_fallback
        
        # Configure exchanges in order of preference
        self.exchanges = self._initialize_exchanges()
        self.current_exchange_idx = 0
        self.exchange = self.exchanges[0]  # Start with primary exchange
        
        self._init_db()
    
    def _initialize_exchanges(self) -> List:
        """Initialize exchanges with Cloudflare bypass and proper headers"""
        exchanges = []
        
        # Common headers to mimic browser requests
        common_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }
        
        # Primary: Binance Futures (with proxy support option)
        try:
            binance = ccxt.binance({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                },
                "timeout": 30000,
                "headers": common_headers,
                # Add proxy if needed: "proxy": "http://your-proxy:port",
            })
            # Force load markets to catch connection issues early
            binance.load_markets()
            exchanges.append(binance)
            print("✅ Binance Futures configured")
        except Exception as e:
            print(f"⚠️ Binance Futures config error: {e}")
        
        # Fallback 1: Binance Spot
        try:
            binance_spot = ccxt.binance({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
                "timeout": 30000,
                "headers": common_headers,
            })
            binance_spot.load_markets()
            exchanges.append(binance_spot)
            print("✅ Binance Spot configured")
        except Exception as e:
            print(f"⚠️ Binance Spot config error: {e}")
        
        # Fallback 2: Bybit (use v5 API properly)
        try:
            bybit = ccxt.bybit({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",
                    "warnOnFetchOpenOrdersWithoutSymbol": False,
                },
                "timeout": 30000,
                "headers": common_headers,
            })
            bybit.load_markets()
            exchanges.append(bybit)
            print("✅ Bybit configured")
        except Exception as e:
            print(f"⚠️ Bybit config error: {e}")
        
        # Fallback 3: OKX
        try:
            okx = ccxt.okx({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "warnOnFetchOpenOrdersWithoutSymbol": False,
                },
                "timeout": 30000,
                "headers": common_headers,
            })
            okx.load_markets()
            exchanges.append(okx)
            print("✅ OKX configured")
        except Exception as e:
            print(f"⚠️ OKX config error: {e}")
        
        # Fallback 4: Kraken (most reliable for spot)
        try:
            kraken = ccxt.kraken({
                "enableRateLimit": True,
                "timeout": 30000,
                "headers": common_headers,
            })
            kraken.load_markets()
            exchanges.append(kraken)
            print("✅ Kraken configured")
        except Exception as e:
            print(f"⚠️ Kraken config error: {e}")
        
        if not exchanges:
            raise Exception("❌ No exchanges could be configured! Check your network/region.")
        
        return exchanges
    def _switch_exchange(self) -> bool:
        """Switch to next available exchange"""
        if not self.use_fallback:
            return False
        
        self.current_exchange_idx += 1
        if self.current_exchange_idx < len(self.exchanges):
            self.exchange = self.exchanges[self.current_exchange_idx]
            exchange_name = self.exchange.name if hasattr(self.exchange, 'name') else 'Unknown'
            print(f"🔄 Switching to fallback exchange: {exchange_name}")
            return True
        return False
    
    def _get_symbol_for_exchange(self, symbol: str, exchange) -> str:
        """Convert symbol format for different exchanges"""
        if hasattr(exchange, 'name'):
            if exchange.name == 'Binance':
                return symbol.replace("/", "")
            elif exchange.name == 'Bybit':
                # Bybit uses BTCUSDT format for linear perpetual
                return symbol.replace("/", "")
            elif exchange.name == 'OKX':
                # OKX uses BTC-USDT-SWAP for perpetual
                base, quote = symbol.split('/')
                return f"{base}-{quote}-SWAP"
            elif exchange.name == 'Kraken':
                # Kraken uses XBT/USD format
                if symbol == "BTC/USDT":
                    return "XBT/USDT"
                return symbol
        return symbol.replace("/", "")
    
    def _get_table_name(self, symbol: str, timeframe: str) -> str:
        """Generate table name from symbol and timeframe"""
        clean_symbol = symbol.replace("/", "_").replace("-", "_")
        return f"ohlcv_{clean_symbol}_{timeframe}"
    
    def _init_db(self):
        """Create DB - tables will be created dynamically as needed"""
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
                sma_50 REAL,
                sma_200 REAL,
                ema_12 REAL,
                ema_26 REAL,
                
                source_exchange TEXT
            )
        """
        )
        conn.commit()
        conn.close()
    
    def _get_last_timestamp(self, symbol: str, timeframe: str):
        """Get last timestamp for specific symbol and timeframe"""
        table_name = self._get_table_name(symbol, timeframe)
        conn = sqlite3.connect(self.db_path)
        
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
    
    def _fetch_with_fallback(self, symbol: str, timeframe: str, since: int = None, limit: int = 1000) -> Optional[List]:
        """Fetch data with automatic exchange fallback"""
        original_idx = self.current_exchange_idx
        
        for attempt in range(len(self.exchanges)):
            try:
                exchange_symbol = self._get_symbol_for_exchange(symbol, self.exchange)
                exchange_name = self.exchange.name if hasattr(self.exchange, 'name') else 'Unknown'
                
                print(f"   📡 Fetching from {exchange_name}...")
                
                ohlcv = self.exchange.fetch_ohlcv(
                    exchange_symbol, timeframe, since=since, limit=limit
                )
                
                if ohlcv:
                    print(f"   ✅ Successfully fetched {len(ohlcv)} candles from {exchange_name}")
                    # Reset to primary exchange for next fetch
                    self.current_exchange_idx = 0
                    self.exchange = self.exchanges[0]
                    return ohlcv
                else:
                    print(f"   ⚠️ No data from {exchange_name}")
                    
            except Exception as e:
                print(f"   ❌ Error with {self.exchange.name if hasattr(self.exchange, 'name') else 'exchange'}: {str(e)[:100]}")
                
                # Try to switch exchange
                if self._switch_exchange():
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    break
        
        # Restore original exchange
        self.current_exchange_idx = original_idx
        self.exchange = self.exchanges[original_idx]
        return None
    
    def _save_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str, source_exchange: str = None):
        """Save data to specific symbol-timeframe table"""
        if df.empty:
            return
        
        self._create_table_if_not_exists(symbol, timeframe)
        
        df = df.copy()
        table_name = self._get_table_name(symbol, timeframe)
        
        # Add source exchange column if provided
        if source_exchange and 'source_exchange' not in df.columns:
            df['source_exchange'] = source_exchange
        
        required_columns = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
             "sma_20", "sma_50", "sma_200", "ema_12", "ema_26"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        db_columns = [
            "timestamp", "open", "high", "low", "close", "volume"
        ] + required_columns
        
        if 'source_exchange' in df.columns:
            db_columns.append('source_exchange')
        
        df = df[db_columns]
        
        conn = sqlite3.connect(self.db_path)
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
        """Fetch new candles and store them in specific table with fallback support"""
        last_ts = self._get_last_timestamp(symbol, timeframe)
        
        if last_ts:
            since = last_ts + 1
            print(
                f"Updating {symbol} {timeframe} from {datetime.utcfromtimestamp(last_ts/1000)}"
            )
        
        all_data = []
        used_exchange = None
        
        while True:
            # Try to fetch with fallback
            ohlcv = self._fetch_with_fallback(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                print(f"❌ All exchanges failed for {symbol} {timeframe}")
                break
            
            # Track which exchange provided the data
            used_exchange = self.exchange.name if hasattr(self.exchange, 'name') else 'Unknown'
            
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"   Fetched {len(ohlcv)} candles → Total: {len(all_data)}")
            
            if len(ohlcv) < limit:
                break
        
        if all_data:
            df = pd.DataFrame(
                all_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = calculate_indicators(df)
            
            if "SMA_20" not in df.columns:
                df["SMA_20"] = df["close"].rolling(window=20).mean()
            if "SMA_50" not in df.columns:
                df["SMA_50"] = df["close"].rolling(window=50).mean()
            if "SMA_200" not in df.columns:
                df["SMA_200"] = df["close"].rolling(window=200).mean()
                
            df.reset_index(inplace=True)
            df["timestamp"] = df["timestamp"].astype("int64") // 10**6
            df["source_exchange"] = used_exchange
            self._save_to_db(df, symbol, timeframe, used_exchange)
            print(f"✅ Stored {len(df)} candles for {symbol} {timeframe} (source: {used_exchange})")
    
    def get_stored_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from specific symbol-timeframe table"""
        table_name = self._get_table_name(symbol, timeframe)
        conn = sqlite3.connect(self.db_path)
        
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
        self, symbols, timeframes=None, start_date="2025-01-01T00:00:00Z", max_retries=3
    ):
        """Main function – download/update multiple symbols & timeframes with fallback"""
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h", "4h","1d"]
        
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        
        for symbol in symbols:
            for tf in timeframes:
                print(f"\n📊 Fetching {symbol} - {tf}")
                
                for retry in range(max_retries):
                    try:
                        self.fetch_and_store(symbol, tf, since=since)
                        break
                    except Exception as e:
                        print(f"   Attempt {retry+1}/{max_retries} failed: {e}")
                        if retry == max_retries - 1:
                            print(f"   ❌ Failed to fetch {symbol} {tf} after {max_retries} attempts")
                            time.sleep(5)
                        else:
                            time.sleep(2)
    
    def get_data_source_info(self, symbol: str, timeframe: str) -> Dict:
        """Get information about which exchanges provided the data"""
        table_name = self._get_table_name(symbol, timeframe)
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT source_exchange, COUNT(*) as count, MIN(timestamp) as first, MAX(timestamp) as last
        FROM {table_name}
        GROUP BY source_exchange
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {}
        
        result = {}
        for _, row in df.iterrows():
            result[row['source_exchange']] = {
                'candles': row['count'],
                'first_candle': pd.to_datetime(row['first'], unit='ms'),
                'last_candle': pd.to_datetime(row['last'], unit='ms')
            }
        
        return result
    
    def close(self):
        pass

# Quick test
if __name__ == "__main__":
    # Test with fallback enabled
    fetcher = DataFetcher(use_fallback=True)
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    # This will automatically use fallback if Binance fails
    fetcher.fetch_multiple_symbols_data(
        symbols, timeframes=["1h"], start_date="2025-01-01T00:00:00Z"
    )
    
    # Check which exchanges provided the data
    for symbol in symbols:
        info = fetcher.get_data_source_info(symbol, "1h")
        print(f"\n📊 {symbol} data sources:")
        for exchange, details in info.items():
            print(f"   {exchange}: {details['candles']} candles")
    
    # Load data as usual (no changes needed to existing code)
    df = fetcher.get_stored_data("BTC/USDT", "1h")
    print(f"\n✅ Loaded BTC/USDT 1h data: {len(df)} rows")