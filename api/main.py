from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from bot_manager import BotManager

app = FastAPI(title="TradeBot Control API")

bot = BotManager()


# ---------- Schemas ----------
class StartRequest(BaseModel):
    mode: Optional[str] = "live"
    args: Optional[List[str]] = None


# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "TradeBot API"}


@app.post("/bot/start")
def start_bot(req: StartRequest):
    args = req.args

    # map mode → CLI args
    if not args:
        if req.mode == "train":
            args = ["--train"]
        elif req.mode == "simple":
            args = ["--simple"]
        elif req.mode == "enhanced":
            args = ["--enhanced"]
        elif req.mode == "validate":
            args = ["--validate"]
        else:
            args = ["--live"]

    return bot.start(args)


@app.post("/bot/stop")
def stop_bot():
    return bot.stop()


@app.get("/bot/status")
def bot_status():
    return bot.status()