import os
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import backend.databricks_db as databricks_db
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from backend.routers import chatbot as chatbot_router

from dotenv import load_dotenv

load_dotenv()


def _warmup_llms() -> None:
    """앱 기동 직후 LLM 엔드포인트를 미리 깨워 사용자 첫 질문의 콜드스타트를 흡수한다.

    측정(2026-06-05): 데워지기 전 첫 질문은 preprocessor LLM이 ~10초, 그 다음 질문은
    report_generator LLM이 ~10초 콜드스타트를 맞았다(웜 상태 총 6.6초 대비 22.7초).
    여기서 두 LLM에 짧은 더미 호출을 미리 날려 그 콜드스타트를 앱 기동 시점으로 옮긴다.
    실패해도 앱은 정상 기동하도록 전부 삼킨다(warm-up은 best-effort 최적화일 뿐).
    """
    try:
        from backend.nodes.preprocessor import _get_llm
        from backend.nodes.report_generator import _get_llm_report
        for getter in (_get_llm, _get_llm_report):
            try:
                getter().invoke("ping")
            except Exception:
                pass  # 개별 LLM warm-up 실패는 무시(다른 것은 계속 시도)
    except Exception:
        pass  # import/초기화 단계 실패도 앱 기동을 막지 않는다


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 기동을 막지 않도록 warm-up은 백그라운드 스레드에서 비동기로 수행한다.
    threading.Thread(target=_warmup_llms, name="llm-warmup", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
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