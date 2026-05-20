import json
import time
import os
from pathlib import Path

LOG_DIR = Path(os.getenv("DEBUG_LOG_DIR", "./debug_archive"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def archive(stage: str, payload: dict) -> None:
    """디버깅용 단계별 입출력을 jsonl 파일과 콘솔에 동시에 기록한다.

    stage 예: "preprocessor.output", "missing_price_search.naver_attempt"
    payload는 직렬화 가능한 dict.
    """
    safe_payload = {}
    for k, v in payload.items():
        try:
            json.dumps(v, ensure_ascii=False)
            safe_payload[k] = v
        except (TypeError, ValueError):
            safe_payload[k] = repr(v)[:500]

    record = {"ts": time.time(), "stage": stage, **safe_payload}
    log_path = LOG_DIR / f"{time.strftime('%Y%m%d')}.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[debug_log] archive 실패: {e}")
    print(f"[{stage}] {json.dumps(safe_payload, ensure_ascii=False)[:300]}")
