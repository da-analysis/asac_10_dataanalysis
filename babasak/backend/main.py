import os
import time
from pathlib import Path

import backend.databricks_db as databricks_db
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from backend.routers import chatbot as chatbot_router

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
app.include_router(chatbot_router.router, prefix="/api/chatbot")

DEBUG_LOG_DIR = Path(os.getenv("DEBUG_LOG_DIR", "./debug_archive"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/ingredients/")
def list_ingredients():
    try:
        live = databricks_db.get_ingredient_prices_live()
        if live:
            return {"items": live, "source": "databricks"}
        return {"items": [], "source": "none", "error": databricks_db.last_error}
    except Exception as exc:
        return {"items": [], "source": "none", "error": str(exc)}


@app.get("/debug/logs/list")
def debug_logs_list():
    """debug_archive 디렉터리에 있는 jsonl 파일 목록."""
    if not DEBUG_LOG_DIR.exists():
        return {"files": [], "dir": str(DEBUG_LOG_DIR.absolute()), "exists": False}
    files = sorted(p.name for p in DEBUG_LOG_DIR.glob("*.jsonl"))
    return {"files": files, "dir": str(DEBUG_LOG_DIR.absolute()), "exists": True}


@app.get("/debug/logs", response_class=PlainTextResponse)
def debug_logs(date: str | None = None, tail: int = 0):
    """debug_archive jsonl 파일 내용 반환.

    Query params:
      date: YYYYMMDD (기본: 오늘)
      tail: 마지막 N 라인만 반환 (0이면 전체)
    """
    target_date = date or time.strftime("%Y%m%d")
    log_path = DEBUG_LOG_DIR / f"{target_date}.jsonl"
    if not log_path.exists():
        return f"# log file not found: {log_path.absolute()}\n"
    text = log_path.read_text(encoding="utf-8")
    if tail > 0:
        lines = text.splitlines()
        text = "\n".join(lines[-tail:])
    return text
