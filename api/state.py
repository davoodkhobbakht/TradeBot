# api/state.py - Single file, ~40 lines
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys, os

# Import your existing config - ZERO changes to original
sys.path.append(os.path.join(os.path.dirname(__file__), "../testnet trader V0.2"))
from config import TRADE_SETTINGS, ML_SETTINGS, RL_SETTINGS, SYMBOL_OPTIMIZED_PARAMS

class AppState:
    """In-memory state for prototype - replace with DB later if needed"""
    
    def __init__(self):
        self.bot_running = False
        self.bot_mode = None
        self.bot_pid = None
        self.bot_logs: List[str] = []
        self.positions: List[Dict] = []  # Mock positions for demo
        self.equity_curve: List[Dict] = []  # Mock chart data
        self.last_backtest_result: Optional[Dict] = None
        self.config_snapshot = {
            "trade": TRADE_SETTINGS,
            "ml": ML_SETTINGS,
            "rl": RL_SETTINGS,
            "symbols": SYMBOL_OPTIMIZED_PARAMS
        }
    
    def add_log(self, message: str):
        self.bot_logs.append(f"[{datetime.now().isoformat()}] {message}")
        if len(self.bot_logs) > 500:  # Keep last 500 lines
            self.bot_logs.pop(0)
    
    def update_positions(self, positions: List[Dict]):
        self.positions = positions
    
    def update_equity(self, data: List[Dict]):
        self.equity_curve = data
    
    def set_backtest_result(self, result: Dict):
        self.last_backtest_result = result

# Singleton instance
state = AppState()