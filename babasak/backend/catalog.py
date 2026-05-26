"""
재료명 카탈로그 + alias 매핑.

목적:
- silver.ingredient.ingredient 테이블의 distinct (재료명, 단위)를 ground truth로 보관.
- 사용자/레시피의 재료명(예: '마늘')을 DB 표준명('깐마늘', '20kg')으로 정규화.
- Genie가 "없다"고 잘못 답해도, 카탈로그에 있으면 후속 단계에서 (재료명, 단위) 명시
  재질의로 보정할 수 있도록 정보를 제공.

설계:
- 카탈로그: statement_execution 1회 호출 + 24h TTL 인메모리 캐싱
- alias: backend/ingredient_aliases.yaml lazy load + mtime 기반 갱신
- resolve_ingredient(): 사용자 입력 → ResolveResult(status, db_name, db_unit, reason)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from backend.debug_log import archive


# ════════════════════════════════════════════════════════════════
# 카탈로그: silver.ingredient.ingredient distinct (재료명, 단위)
# ════════════════════════════════════════════════════════════════

_CATALOG_SQL = """
SELECT DISTINCT `재료명`, `단위`
FROM silver.ingredient.ingredient
WHERE `재료명` IS NOT NULL AND `단위` IS NOT NULL
"""

_CATALOG_TTL_SECONDS = 24 * 60 * 60  # 24h

_catalog_cache: dict[str, set[str]] | None = None
_catalog_loaded_at: float = 0.0


def _get_warehouse_id(w: WorkspaceClient) -> str | None:
    """databricks_db.py의 동일 헬퍼를 카탈로그용으로 재현."""
    wh_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if wh_id:
        return wh_id
    for wh in w.warehouses.list():
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            return wh.id
    warehouses = list(w.warehouses.list())
    return warehouses[0].id if warehouses else None


def _load_catalog_from_db() -> dict[str, set[str]]:
    """SQL 1회 호출로 {재료명: {단위, ...}} 형태 반환."""
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    w = WorkspaceClient(host=host, token=token) if host and token else WorkspaceClient()

    warehouse_id = _get_warehouse_id(w)
    if not warehouse_id:
        raise RuntimeError("warehouse를 찾을 수 없음 (DATABRICKS_WAREHOUSE_ID 미설정)")

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=_CATALOG_SQL,
        wait_timeout="50s",
    )
    if not resp.status or resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status and resp.status.error else "unknown"
        raise RuntimeError(f"카탈로그 SQL 실행 실패: {err}")

    catalog: dict[str, set[str]] = {}
    for row in (resp.result.data_array or []):
        name = (row[0] or "").strip()
        unit = (row[1] or "").strip()
        if not name or not unit:
            continue
        catalog.setdefault(name, set()).add(unit)
    return catalog


def get_catalog(force_reload: bool = False) -> dict[str, set[str]]:
    """카탈로그 반환. TTL 만료 또는 force_reload 시 재로드.

    실패해도 빈 dict 반환 (앱이 죽지 않도록). 그 경우 resolve_ingredient는
    모든 입력을 'not_in_catalog'로 처리하여 기존 동작을 유지함.
    """
    global _catalog_cache, _catalog_loaded_at

    now = time.time()
    expired = (now - _catalog_loaded_at) > _CATALOG_TTL_SECONDS
    if _catalog_cache is not None and not expired and not force_reload:
        return _catalog_cache

    try:
        _catalog_cache = _load_catalog_from_db()
        _catalog_loaded_at = now
        archive("catalog.loaded", {
            "num_ingredients": len(_catalog_cache),
            "sample": list(_catalog_cache.keys())[:10],
        })
    except Exception as e:
        archive("catalog.load_failed", {"error": str(e)})
        if _catalog_cache is None:
            _catalog_cache = {}

    return _catalog_cache


# ════════════════════════════════════════════════════════════════
# Alias 테이블: backend/ingredient_aliases.yaml
# ════════════════════════════════════════════════════════════════

_ALIAS_YAML_PATH = Path(__file__).parent / "ingredient_aliases.yaml"

_alias_cache: dict[str, dict | None] | None = None
_alias_mtime: float = 0.0


def _load_alias_yaml() -> dict[str, dict | None]:
    """YAML 파일을 읽어 {입력명: {db_name, db_unit} | None} 형태로 반환.

    값이 None이면 ambiguous(자동 매칭 금지) 표시.
    파일이 없거나 파싱 실패해도 빈 dict로 graceful 처리.
    """
    if not _ALIAS_YAML_PATH.exists():
        archive("catalog.alias_yaml_missing", {"path": str(_ALIAS_YAML_PATH)})
        return {}

    try:
        import yaml  # pyyaml — requirements.txt 추가 필요
    except ImportError:
        archive("catalog.alias_yaml_no_pyyaml", {})
        return {}

    try:
        with open(_ALIAS_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        aliases = data.get("aliases") or {}
        if not isinstance(aliases, dict):
            archive("catalog.alias_yaml_bad_shape", {"type": type(aliases).__name__})
            return {}
        # 값 검증: dict {db_name, db_unit} 또는 None만 허용
        cleaned: dict[str, dict | None] = {}
        for key, val in aliases.items():
            key = str(key).strip()
            if val is None:
                cleaned[key] = None
            elif isinstance(val, dict) and "db_name" in val and "db_unit" in val:
                cleaned[key] = {
                    "db_name": str(val["db_name"]).strip(),
                    "db_unit": str(val["db_unit"]).strip(),
                }
            # 그 외 형식은 무시
        return cleaned
    except Exception as e:
        archive("catalog.alias_yaml_error", {"error": str(e)})
        return {}


def get_alias_table() -> dict[str, dict | None]:
    """alias 테이블 반환. mtime 변경 감지 시 자동 재로드."""
    global _alias_cache, _alias_mtime

    if _ALIAS_YAML_PATH.exists():
        current_mtime = _ALIAS_YAML_PATH.stat().st_mtime
    else:
        current_mtime = 0.0

    if _alias_cache is None or current_mtime != _alias_mtime:
        _alias_cache = _load_alias_yaml()
        _alias_mtime = current_mtime
        archive("catalog.alias_loaded", {
            "num_entries": len(_alias_cache),
            "ambiguous_count": sum(1 for v in _alias_cache.values() if v is None),
        })

    return _alias_cache


# ════════════════════════════════════════════════════════════════
# Resolve: 입력 재료명 → DB 표준명/단위
# ════════════════════════════════════════════════════════════════

ResolveStatus = Literal["matched", "ambiguous", "not_in_catalog", "unmapped"]


@dataclass
class ResolveResult:
    """재료명 해석 결과.

    status:
      - 'matched'         : alias에 매핑 있고 카탈로그에 (db_name, db_unit) 존재
                            → Genie 1차 호출 + 누락 시 (db_name, db_unit) 재질의 대상
      - 'ambiguous'       : alias 값이 None — 자동 추측 금지 (예: '돼지고기', '고추')
                            → Genie 1차는 원본 이름으로 시도하되, 못 찾으면 바로 네이버
      - 'not_in_catalog'  : alias 매핑은 있는데 카탈로그에 db_name/db_unit이 없음
                            → 네이버 폴백 대상 (DB에 정말 없음)
      - 'unmapped'        : alias에 키 자체가 없음 (운영 중 추가 필요)
                            → Genie 1차는 원본 이름으로 시도, 못 찾으면 네이버
    """
    input_name: str
    status: ResolveStatus
    db_name: str | None
    db_unit: str | None
    reason: str

    @property
    def has_catalog_target(self) -> bool:
        """'카탈로그에 있다고 확신할 수 있는' 케이스인지. Genie 재질의 대상 판정용."""
        return self.status == "matched"


def resolve_ingredient(name: str) -> ResolveResult:
    """입력 재료명을 카탈로그+alias로 해석.

    호출 순서:
      1. alias에 키가 있는가?
         - 값이 dict: db_name이 카탈로그에 있으면 'matched', 없으면 'not_in_catalog'
         - 값이 None: 'ambiguous'
      2. alias에 키가 없으면: 카탈로그 자체에 정확명이 있는지 확인
         - 있으면 'matched' (단위는 임의 1개 선택)
         - 없으면 'unmapped'
    """
    name = (name or "").strip()
    if not name:
        return ResolveResult(name, "unmapped", None, None, "empty_input")

    alias_table = get_alias_table()
    catalog = get_catalog()

    # 1. alias 우선
    if name in alias_table:
        alias_val = alias_table[name]
        if alias_val is None:
            return ResolveResult(name, "ambiguous", None, None, "alias_explicit_ambiguous")
        db_name = alias_val["db_name"]
        db_unit = alias_val["db_unit"]
        catalog_units = catalog.get(db_name, set())
        if db_unit in catalog_units:
            return ResolveResult(name, "matched", db_name, db_unit, "alias_hit_catalog")
        return ResolveResult(
            name, "not_in_catalog", db_name, db_unit,
            f"alias_maps_to_{db_name}/{db_unit}_but_catalog_has_{sorted(catalog_units) or 'none'}",
        )

    # 2. alias 미등록 — 카탈로그 정확명 매칭만 시도 (부분/유사 매칭은 안 함)
    if name in catalog:
        units = catalog[name]
        if units:
            unit = sorted(units)[0]
            return ResolveResult(name, "matched", name, unit, "direct_catalog_hit")

    return ResolveResult(name, "unmapped", None, None, "no_alias_no_direct_catalog")


def resolve_many(names: list[str]) -> list[ResolveResult]:
    """여러 재료를 한 번에 해석. 중복은 첫 등장만 유지 (호출자가 묶을 때 dedup하므로)."""
    seen: set[str] = set()
    results: list[ResolveResult] = []
    for n in names:
        key = (n or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(resolve_ingredient(key))
    return results
