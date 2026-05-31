
from fastapi import FastAPI, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio, json, sys, os

# Add your project to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from config import TRADE_SETTINGS, ML_SETTINGS, RL_SETTINGS, SYMBOL_OPTIMIZED_PARAMS
from api.bot_manager import BotManager
from api.state import state

app = FastAPI(title="TradeBot Demo API", version="0.2.0")

# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = BotManager()

# ---------- Schemas ----------
class StartRequest(BaseModel):
    mode: Optional[str] = "live"
    args: Optional[List[str]] = None

class ConfigUpdate(BaseModel):
    trade: Optional[Dict] = None
    ml: Optional[Dict] = None
    rl: Optional[Dict] = None
    symbols: Optional[Dict] = None

# ---------- Bot Control (existing + enhanced) ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "TradeBot Demo API"}

@app.post("/bot/start")
async def start_bot(req: StartRequest):
    args = req.args
    if not args:
        mode_map = {
            "train": ["--train"], "simple": ["--simple"],
            "enhanced": ["--enhanced"], "validate": ["--validate"],
            "live": ["--live"]
        }
        args = mode_map.get(req.mode, ["--live"])
    
    result = bot.start(args)
    if result["status"] == "started":
        state.bot_running = True
        state.bot_mode = req.mode
        state.bot_pid = result.get("pid")
        # Start log reader in background
        asyncio.create_task(_stream_bot_logs())
    return result

@app.post("/bot/stop")
def stop_bot():
    result = bot.stop()
    if result["status"] == "stopped":
        state.bot_running = False
        state.bot_pid = None
    return result

@app.get("/bot/status")
def bot_status():
    return {
        "status": "running" if state.bot_running else "stopped",
        "mode": state.bot_mode,
        "pid": state.bot_pid,
        "uptime": "N/A"  # Add if needed
    }

# ---------- Config API ----------
@app.get("/config")
def get_config():
    return state.config_snapshot

@app.put("/config")
def update_config(update: ConfigUpdate):
    if update.trade:
        state.config_snapshot["trade"].update(update.trade)
    if update.ml:
        state.config_snapshot["ml"].update(update.ml)
    if update.rl:
        state.config_snapshot["rl"].update(update.rl)
    if update.symbols:
        state.config_snapshot["symbols"].update(update.symbols)
    # Note: Changes apply to next run (config.py would need reload for live)
    return {"status": "updated", "config": state.config_snapshot}

# ---------- Positions API (Mock for Demo) ----------
@app.get("/positions")
def get_positions():
    # Return mock data - replace with real position tracking later
    return state.positions or [
        {"symbol": "BTC/USDT", "side": "LONG", "entry": 65000, "current": 65800, "pnl_pct": 1.23},
        {"symbol": "ETH/USDT", "side": "SHORT", "entry": 3200, "current": 3150, "pnl_pct": 1.56}
    ]

# ---------- Backtest API ----------
@app.post("/backtest/simple")
async def run_simple_backtest(background_tasks: BackgroundTasks):
    """Run simple backtest via trading_engine"""
    
    def _run():
        from trading_engine import simple_backtest
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        result = simple_backtest(symbols, logger_callback=state.add_log)
        state.set_backtest_result({"type": "simple", "data": result})
        state.add_log("✅ Simple backtest completed")
    
    background_tasks.add_task(_run)
    return {"job_id": "simple_bt", "status": "queued"}


@app.post("/backtest/enhanced")
async def run_enhanced_backtest(background_tasks: BackgroundTasks):
    """Run enhanced backtest via trading_engine"""
    
    def _run():
        from trading_engine import enhanced_backtest
        from ml.base_ml import MLModelManager
        
        ml_manager = MLModelManager()
        if not ml_manager.load_models():
            state.add_log("❌ No ML models found")
            state.set_backtest_result({"type": "enhanced", "error": "Models not found"})
            return
        
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        result = enhanced_backtest(symbols, ml_manager.models, logger_callback=state.add_log)
        state.set_backtest_result({"type": "enhanced", "data": result})
        state.add_log("✅ Enhanced backtest completed")
    
    background_tasks.add_task(_run)
    return {"job_id": "enhanced_bt", "status": "queued"}
@app.get("/backtest/result")
def get_backtest_result():
    return state.last_backtest_result or {"status": "no_result_yet"}

# ---------- Reports API ----------
@app.get("/reports/latest")
def get_latest_report():
    # Mock report - replace with real metrics from backtest/performance.py
    return {
        "return_pct": 18.4,
        "win_rate": 61,
        "max_drawdown": 4.2,
        "total_trades": 148,
        "profit_factor": 1.85,
        "sharpe_ratio": 1.32
    }

# ---------- Equity Chart API ----------
@app.get("/chart/equity")
def get_equity_chart(days: int = 30):
    # Mock data - replace with real equity_curve from backtester
    import random
    from datetime import datetime, timedelta
    
    base = 1000
    data = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d")
        base *= (1 + random.uniform(-0.02, 0.03))  # Random walk for demo
        data.append({"time": date, "equity": round(base, 2)})
    
    return data

# ---------- Logs API ----------
@app.get("/logs")
def get_logs(lines: int = 100):
    return state.bot_logs[-lines:]

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send new logs every 2 seconds
            logs = state.bot_logs[-50:]  # Last 50 lines
            await websocket.send_json({"type": "logs", "data": logs})
            await asyncio.sleep(2)
    except:
        pass  # Client disconnected

# ---------- Background Task: Stream Bot Logs ----------
async def _stream_bot_logs():
    """Read stdout from bot process and add to state"""
    if not bot.process or not bot.process.stdout:
        return
    
    while bot.process.poll() is None:  # While running
        line = bot.process.stdout.readline()
        if line:
            state.add_log(line.strip())
        await asyncio.sleep(0.1)  # Non-blocking

# Auto-docs at /docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)