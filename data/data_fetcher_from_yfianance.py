# -*- coding: utf-8 -*-
"""
Data Fetcher with Yahoo Finance as Primary Source & Auto Indicator Generation
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    logger.error("ccxt not installed. Run: pip install ccxt")
    ccxt = None

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


class DataFetcher:
    """
    Fetches historical data with Yahoo Finance as PRIMARY source.
    Automatically calculates technical indicators required by ML models.
    """

    def __init__(self, data_dir: str = "data/cache", use_exchanges: bool = False):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.exchanges = []
        if use_exchanges and ccxt:
            print("🔄 Initializing CCXT exchanges as fallback...")
            self.exchanges = self._initialize_exchanges()
        else:
            print("✅ Using Yahoo Finance as primary data source")

        self.data_cache = {}

    def _initialize_exchanges(self) -> List:
        exchanges = []
        common_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        try:
            bybit = ccxt.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "defaultSubType": "linear"},
                "timeout": 30000, "headers": common_headers,
            })
            bybit.load_markets()
            exchanges.append(bybit)
<<<<<<< HEAD
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
=======
        except Exception: pass
>>>>>>> aea1e5a (data fetching from yfinance)

        try:
            binance = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
                "timeout": 30000, "headers": common_headers,
            })
            binance.load_markets()
            exchanges.append(binance)
        except Exception: pass

        if not exchanges:
            print("⚠️ No CCXT exchanges available. Yahoo Finance only.")
        return exchanges

    def _convert_symbol_for_yahoo(self, symbol: str) -> str:
        base = symbol.split('/')[0]
        return f"{base}-USD"

    def _get_yahoo_period_and_interval(self, timeframe: str) -> tuple:
        mapping = {
            '1m': ('7d', '1m'), '5m': ('60d', '5m'), '15m': ('60d', '15m'),
            '30m': ('60d', '30m'), '1h': ('730d', '1h'),
            '4h': ('730d', '1h'), '1d': ('max', '1d'), '1w': ('max', '1wk'),
        }
        return mapping.get(timeframe, ('max', '1d'))

    def _parse_date(self, date_str) -> str:
        if not date_str: return None
        if isinstance(date_str, str):
            return date_str.replace('Z', '').split('T')[0]
        return str(date_str)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators required by ML models, RL, and Validation"""
        df = df.copy()

        # 1. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Alias for base_strategy.py
        df['Signal_Line'] = df['MACD_Signal']

        # 3. Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # 4. Bollinger Bands (20, 2)
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # 5. ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        # Alias for multi_strategy.py
        df['ATR'] = df['atr_14']

        # 6. ADX (14)
        high, low, close = df['high'], df['low'], df['close']
        tr1 = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        up = high.diff()
        down = low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / tr1.rolling(14).mean())
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / tr1.rolling(14).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()

        # 7. Stochastic Oscillator (14, 3) - 🌟 ADDED FOR MOMENTUM STRATEGY
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Fill NaN values created by rolling windows
        df = df.ffill().bfill()

        return df

    def fetch_from_yahoo_finance(self, symbol: str, timeframe: str = "1d", **kwargs) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            print("⚠️ yfinance is not installed. Cannot fetch data.")
            return pd.DataFrame()

        try:
            yahoo_symbol = self._convert_symbol_for_yahoo(symbol)
            start_date = kwargs.get('start_date', None)
            clean_start_date = self._parse_date(start_date)

            ticker = yf.Ticker(yahoo_symbol)
            yf_interval = '1h' if timeframe == '4h' else timeframe

            yf_limits = {
                '1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60,
                '60m': 730, '1h': 730, '1d': 99999, '1wk': 99999
            }
            max_days = yf_limits.get(yf_interval, 99999)

            use_period = False
            period_to_use = "max"

            if clean_start_date:
                try:
                    start_dt = pd.to_datetime(clean_start_date)
                    if start_dt.tzinfo:
                        start_dt = start_dt.tz_localize(None)
                    now = pd.Timestamp.now()
                    days_diff = (now - start_dt).days

                    if days_diff > max_days:
                        use_period = True
                        if yf_interval in ['1m']: period_to_use = '7d'
                        elif yf_interval in ['5m', '15m', '30m']: period_to_use = '60d'
                        elif yf_interval in ['1h']: period_to_use = '730d'
                        else: period_to_use = 'max'
                except Exception:
                    pass

            if use_period:
                print(f"⚠️ Yahoo Finance only keeps {period_to_use} of {yf_interval} data. Fetching maximum available...")
                df = ticker.history(period=period_to_use, interval=yf_interval)
            elif clean_start_date:
                print(f"📥 Fetching {yahoo_symbol} from Yahoo Finance (start={clean_start_date})...")
                df = ticker.history(start=clean_start_date, interval=yf_interval)
            else:
                period, interval = self._get_yahoo_period_and_interval(timeframe)
                print(f"📥 Fetching {yahoo_symbol} from Yahoo Finance (period={period}, interval={interval})...")
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"⚠️ No data from Yahoo Finance for {yahoo_symbol}")
                return pd.DataFrame()

            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = 'timestamp'

            if timeframe == '4h':
                df = df.resample('4h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()

            # 🌟 CRITICAL FIX: Add technical indicators before caching!
            df = self.add_technical_indicators(df)

            print(f"✅ Fetched {len(df)} candles with indicators for {symbol}")
            return df

        except Exception as e:
            print(f"⚠️ Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 1000, max_retries: int = 3, **kwargs) -> pd.DataFrame:
        cached = self.get_stored_data(symbol, timeframe)
        # Check if cached data has indicators (e.g., 'RSI' column)
        if not cached.empty and 'RSI' in cached.columns and len(cached) >= limit * 0.8:
            print(f"💾 Using cached data for {symbol} ({timeframe}) - {len(cached)} candles")
            return cached.tail(limit)

        print(f"🌐 Fetching data for {symbol} ({timeframe})...")

        df = self.fetch_from_yahoo_finance(symbol, timeframe, **kwargs)
        if not df.empty:
            self._store_data(symbol, timeframe, df)
            return df

        if not self.exchanges:
            self.exchanges = self._initialize_exchanges()

        if self.exchanges:
            for exchange in self.exchanges:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol.replace('/', ''), timeframe, limit=limit)
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        df.index.name = 'timestamp'
                        df = self.add_technical_indicators(df) # Add indicators for exchange data too
                        self._store_data(symbol, timeframe, df)
                        return df
                except Exception:
                    continue

        raise Exception(f"❌ Could not fetch data for {symbol} ({timeframe}) from any source")

    def fetch_multiple_symbols_data(self, symbols: List[str], timeframes: List[str] = None, limit: int = 1000, **kwargs):
        if timeframes is None:
            timeframes = ["1d", "4h", "1h"]

        if "1d" not in timeframes:
            timeframes.append("1d")
            print("ℹ️ Automatically added '1d' to timeframes (required for ML training)")

        print(f"\n📥 Fetching multi-timeframe historical data for {len(symbols)} symbols...")
        print(f"   Timeframes to fetch: {timeframes}")

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    self.fetch_ohlcv(symbol, timeframe, limit, **kwargs)
                    time.sleep(1)
                except Exception as e:
                    print(f"❌ Failed to fetch data for {symbol} ({timeframe}): {e}")

    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return os.path.join(self.data_dir, f"{safe_symbol}_{timeframe}.csv")

    def _store_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        try:
            path = self._get_cache_path(symbol, timeframe)
            df.to_csv(path)
            print(f"💾 Cached {len(df)} candles with indicators for {symbol} ({timeframe})")
        except Exception as e:
            print(f"⚠️ Failed to cache data for {symbol}: {e}")

    def get_stored_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            path = self._get_cache_path(symbol, timeframe)
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index.name = 'timestamp'
                return df
        except Exception as e:
            print(f"⚠️ Failed to load cached data for {symbol}: {e}")
        return pd.DataFrame()
