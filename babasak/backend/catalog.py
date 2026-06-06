from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from backend.debug_log import archive



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


def _get_ws_client() -> WorkspaceClient:
    """WorkspaceClient를 환경변수 기반으로 생성."""
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    return WorkspaceClient(host=host, token=token) if host and token else WorkspaceClient()


# 1차 카탈로그: silver.ingredient.ingredient distinct (재료명, 단위)

_CATALOG_SQL = """
SELECT DISTINCT `재료명`, `단위`
FROM silver.ingredient.ingredient
WHERE `재료명` IS NOT NULL AND `단위` IS NOT NULL
"""

_CATALOG_TTL_SECONDS = 24 * 60 * 60  # 24h

_catalog_cache: dict[str, set[str]] | None = None
_catalog_loaded_at: float = 0.0


def _load_catalog_from_db() -> dict[str, set[str]]:
    """SQL 1회 호출로 {재료명: {단위, ...}} 형태 반환."""
    w = _get_ws_client()
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
    """KAMIS 카탈로그 반환. TTL 만료 또는 force_reload 시 재로드."""
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


# 매핑 테이블: silver.ingredient.ingredient_mapping (recipe→KAMIS 연결)

_MAPPING_SQL = """
SELECT std_name, kamis_name, kamis_unit, mapping_method, confidence
FROM silver.ingredient.ingredient_mapping
WHERE has_kamis_mapping = true AND confidence >= 0.5
"""

_MAPPING_TTL_SECONDS = 24 * 60 * 60  # 24h

_mapping_cache: dict[str, dict] | None = None
_mapping_loaded_at: float = 0.0


def _load_mapping_from_db() -> dict[str, dict]:
    """매핑 테이블 로드 → {std_name: {kamis_name, kamis_unit, method, confidence}}."""
    w = _get_ws_client()
    warehouse_id = _get_warehouse_id(w)
    if not warehouse_id:
        raise RuntimeError("warehouse를 찾을 수 없음 (DATABRICKS_WAREHOUSE_ID 미설정)")

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=_MAPPING_SQL,
        wait_timeout="50s",
    )
    if not resp.status or resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status and resp.status.error else "unknown"
        raise RuntimeError(f"매핑 테이블 SQL 실행 실패: {err}")

    mapping: dict[str, dict] = {}
    for row in (resp.result.data_array or []):
        std_name = (row[0] or "").strip()
        kamis_name = (row[1] or "").strip()
        kamis_unit = (row[2] or "").strip()
        if not std_name or not kamis_name or not kamis_unit:
            continue
        # 동일 std_name이 여러 행일 수 있으므로 첫 번째(highest confidence)만 유지
        if std_name in mapping:
            continue
        try:
            confidence = float(row[4]) if row[4] is not None else 0.5
        except (TypeError, ValueError):
            confidence = 0.5
        mapping[std_name] = {
            "kamis_name": kamis_name,
            "kamis_unit": kamis_unit,
            "method": (row[3] or "").strip(),
            "confidence": confidence,
        }
    return mapping


def get_mapping_table(force_reload: bool = False) -> dict[str, dict]:
    """매핑 테이블 반환. TTL 만료 또는 force_reload 시 재로드."""
    global _mapping_cache, _mapping_loaded_at

    now = time.time()
    expired = (now - _mapping_loaded_at) > _MAPPING_TTL_SECONDS
    if _mapping_cache is not None and not expired and not force_reload:
        return _mapping_cache

    try:
        _mapping_cache = _load_mapping_from_db()
        _mapping_loaded_at = now
        archive("catalog.mapping_loaded", {
            "num_mappings": len(_mapping_cache),
            "sample": list(_mapping_cache.keys())[:10],
        })
    except Exception as e:
        archive("catalog.mapping_load_failed", {"error": str(e)})
        if _mapping_cache is None:
            _mapping_cache = {}

    return _mapping_cache


# 2차 카탈로그: silver.ingredient.ingredient_recipe (B2B 유통가)

_RECIPE_CATALOG_SQL = """
SELECT
    std_name,
    `상품명`,
    `가격`,
    `단위`,
    `단위_수치`,
    `단위_문자`,
    lv1,
    lv2,
    matched_any_reference,
    agro_category_ref,
    kca_category_ref
FROM silver.ingredient.ingredient_recipe
WHERE `가격` IS NOT NULL AND std_name IS NOT NULL
"""

_RECIPE_TTL_SECONDS = 24 * 60 * 60  # 24h

_recipe_cache: dict[str, dict] | None = None
_recipe_loaded_at: float = 0.0


@dataclass
class RecipeIngredientInfo:
    """ingredient_recipe 테이블의 재료 정보."""
    std_name: str
    product_name: str          # 상품명
    price: int                 # B2B 유통가 (원)
    unit: str                  # 단위 (예: "200g", "1kg")
    unit_numeric: float | None  # 단위_수치
    unit_text: str | None      # 단위_문자 (g, kg, ml 등)
    lv1: str | None            # 대분류
    lv2: str | None            # 중분류
    matched_any_reference: bool  # KAMIS 매칭 여부
    agro_category_ref: str | None
    kca_category_ref: float | None

    @property
    def price_per_kg(self) -> int | None:
        """kg당 단가 환산. 환산 불가하면 None."""
        if not self.unit_numeric or self.unit_numeric <= 0:
            return None
        unit_lower = (self.unit_text or "").lower().strip()
        if unit_lower in ("g", "그램"):
            return int(self.price * 1000 / self.unit_numeric)
        elif unit_lower in ("kg", "킬로그램"):
            return int(self.price / self.unit_numeric)
        elif unit_lower in ("ml", "밀리리터"):
            return int(self.price * 1000 / self.unit_numeric)
        elif unit_lower in ("l", "리터"):
            return int(self.price / self.unit_numeric)
        return None


_RECIPE_WEIGHT_UNIT_TEXTS = ("g", "그램", "kg", "킬로그램", "ml", "밀리리터", "l", "리터")


def _recipe_row_convertible(entry: dict) -> bool:
    """이 B2B 행이 kg당 단가로 환산 가능한지(무게/부피 단위 + 단위_수치 존재)."""
    return bool(entry.get("unit_numeric")) and (entry.get("unit_text") or "").lower() in _RECIPE_WEIGHT_UNIT_TEXTS


def _load_recipe_catalog_from_db() -> dict[str, dict]:
    """ingredient_recipe 전체 로드 → {std_name: info dict}."""
    w = _get_ws_client()
    warehouse_id = _get_warehouse_id(w)
    if not warehouse_id:
        raise RuntimeError("warehouse를 찾을 수 없음 (DATABRICKS_WAREHOUSE_ID 미설정)")

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=_RECIPE_CATALOG_SQL,
        wait_timeout="50s",
    )
    if not resp.status or resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status and resp.status.error else "unknown"
        raise RuntimeError(f"recipe 카탈로그 SQL 실행 실패: {err}")

    recipe_catalog: dict[str, dict] = {}
    for row in (resp.result.data_array or []):
        std_name = (row[0] or "").strip()
        if not std_name:
            continue
        try:
            price = int(float(row[2])) if row[2] is not None else None
        except (TypeError, ValueError):
            price = None
        if price is None or price <= 0:
            continue
        try:
            unit_numeric = float(row[4]) if row[4] is not None else None
        except (TypeError, ValueError):
            unit_numeric = None
        try:
            kca_ref = float(row[10]) if row[10] is not None else None
        except (TypeError, ValueError):
            kca_ref = None

        entry = {
            "std_name": std_name,
            "product_name": (row[1] or "").strip(),
            "price": price,
            "unit": (row[3] or "").strip(),
            "unit_numeric": unit_numeric,
            "unit_text": (row[5] or "").strip() if row[5] else None,
            "lv1": (row[6] or "").strip() if row[6] else None,
            "lv2": (row[7] or "").strip() if row[7] else None,
            "matched_any_reference": str(row[8]).lower() == "true" if row[8] else False,
            "agro_category_ref": (row[9] or "").strip() if row[9] else None,
            "kca_category_ref": kca_ref,
        }
        # 같은 std_name이 여러 행이면 'kg 환산 가능한 행'을 우선 유지한다.
        #   (개/봉 단위 행이 먼저 잡혀 가격이 죽는 것 방지 — 시래기 등)
        existing = recipe_catalog.get(std_name)
        if existing is None or (_recipe_row_convertible(entry) and not _recipe_row_convertible(existing)):
            recipe_catalog[std_name] = entry
    return recipe_catalog


def get_recipe_catalog(force_reload: bool = False) -> dict[str, dict]:
    """recipe 카탈로그 반환. TTL 만료 또는 force_reload 시 재로드."""
    global _recipe_cache, _recipe_loaded_at

    now = time.time()
    expired = (now - _recipe_loaded_at) > _RECIPE_TTL_SECONDS
    if _recipe_cache is not None and not expired and not force_reload:
        return _recipe_cache

    try:
        _recipe_cache = _load_recipe_catalog_from_db()
        _recipe_loaded_at = now
        archive("catalog.recipe_loaded", {
            "num_ingredients": len(_recipe_cache),
            "sample": list(_recipe_cache.keys())[:10],
        })
    except Exception as e:
        archive("catalog.recipe_load_failed", {"error": str(e)})
        if _recipe_cache is None:
            _recipe_cache = {}

    return _recipe_cache


# 재료명 매칭 로직: 정확 → 공백정규화 → 접두사제거 → 부분매칭 → 접미사 매칭

_COOKING_PREFIXES = (
    "잘게썬", "얇게썬", "굵게썬", "채썬", "깍둑썬",
    "다진", "냉동", "해동", "볶은", "삶은", "데친", "구운", "튀긴", "말린",
    "간", "생", "건", "훈제",
)

_TRAILING_QUANTITY_RE = re.compile(r"[\d½¼¾]+\s*(?:개|쪽|대|장|줄기|마리|g|kg|ml|L|큰술|작은술|컵|T|t)?\s*$")


def _normalize_name(name: str) -> str:
    """매칭 전처리: 공백 제거."""
    return name.replace(" ", "").strip()


def _strip_cooking_prefix(name: str) -> str | None:
    """조리 접두사 제거. 제거 후 2글자 이상이면 반환, 아니면 None."""
    for prefix in _COOKING_PREFIXES:
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            if len(remainder) >= 2:
                return remainder
    return None


def _strip_trailing_quantity(name: str) -> str | None:
    """뒤쪽 수량/단위 제거. 예: '양파1개' → '양파'."""
    result = _TRAILING_QUANTITY_RE.sub("", name).strip()
    if result and result != name and len(result) >= 2:
        return result
    return None


_WEIGHT_UNIT_RE = re.compile(r"^\d+(?:\.\d+)?(?:kg|g|ml|l)$", re.IGNORECASE)


def _is_weight_unit(unit: str) -> bool:
    """단위가 무게/부피(kg/g/ml/l)라 kg당 단가로 바로 환산 가능한지."""
    return bool(_WEIGHT_UNIT_RE.match((unit or "").replace(" ", "")))


def _unit_to_grams(unit: str) -> float:
    """무게/부피 단위 문자열을 g(또는 ml)로 환산. 환산 불가하면 0."""
    m = _WEIGHT_UNIT_RE.match((unit or "").replace(" ", ""))
    if not m:
        return 0.0
    s = (unit or "").replace(" ", "").lower()
    num = float(re.match(r"\d+(?:\.\d+)?", s).group())
    return num * 1000 if (s.endswith("kg") or s.endswith("l")) else num


def _best_unit(units: set[str]) -> str:
    """KAMIS 단위 후보 중 무게단위(kg/g)를 개수단위(개/장)보다 우선 선택.

    개수단위 도매가는 팩 크기에 따라 단가가 들쭉날쭉(계란 10개 608원/개 vs 30개 205원/개)
    이라 kg 환산 신뢰도가 낮다. 무게단위가 있으면 그걸 고르되, 소포장(100g)은 도매가가
    부풀려지므로 실제 중량이 가장 큰 단위(2kg > 100g)를 우선해 정확도를 높인다.
    """
    weights = [u for u in units if _is_weight_unit(u)]
    if weights:
        return max(weights, key=_unit_to_grams)
    return sorted(units)[0]


def _lookup_recipe(name: str, recipe_catalog: dict[str, dict]) -> dict | None:
    """recipe 카탈로그에서 재료를 찾는 내부 헬퍼.

    5단계 매칭 전략 (높은 정확도 순):
      1. 정확 매칭
      2. 공백 제거 후 정확 매칭
      3. 조리 접두사 제거 후 매칭
      4. 수량 접미사 제거 후 매칭
      5. 부분 포함 매칭 (min 2글자 겹침 + 50% 이상 겹침 비율)
    """
    if not recipe_catalog or not name:
        return None

    # ── 1. 정확 매칭 ──
    if name in recipe_catalog:
        return recipe_catalog[name]

    # ── 2. 공백 제거 후 정확 매칭 ──
    normalized = _normalize_name(name)
    if normalized != name and normalized in recipe_catalog:
        return recipe_catalog[normalized]

    for std_name, info in recipe_catalog.items():
        if _normalize_name(std_name) == normalized:
            return info

    # ── 3. 조리 접두사 제거 후 매칭 ──
    stripped = _strip_cooking_prefix(normalized)
    if stripped:
        if stripped in recipe_catalog:
            return recipe_catalog[stripped]
        for std_name, info in recipe_catalog.items():
            if _normalize_name(std_name) == stripped:
                return info

    # ── 4. 수량 접미사 제거 후 매칭 ──
    qty_stripped = _strip_trailing_quantity(normalized)
    if qty_stripped:
        if qty_stripped in recipe_catalog:
            return recipe_catalog[qty_stripped]
        for std_name, info in recipe_catalog.items():
            if _normalize_name(std_name) == qty_stripped:
                return info

    # ── 5. 부분 포함 매칭 (완화된 조건) ──
    candidates = []
    for std_name, info in recipe_catalog.items():
        std_norm = _normalize_name(std_name)
        if normalized in std_norm or std_norm in normalized:
            overlap = min(len(normalized), len(std_norm))
            longer = max(len(normalized), len(std_norm))
            if overlap >= 2 and overlap / longer >= 0.5:
                diff = abs(len(normalized) - len(std_norm))
                score = diff - (overlap / longer)
                candidates.append((score, std_name, info))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2]

    return None


def get_recipe_price(name: str) -> RecipeIngredientInfo | None:
    """재료명으로 recipe 카탈로그에서 B2B 가격 정보 조회."""
    recipe_catalog = get_recipe_catalog()
    info = _lookup_recipe(name, recipe_catalog)
    if info:
        return RecipeIngredientInfo(**info)
    return None


# Alias 테이블: backend/ingredient_aliases.yaml

_ALIAS_YAML_PATH = Path(__file__).parent / "ingredient_aliases.yaml"

_alias_cache: dict[str, dict | None] | None = None
_alias_mtime: float = 0.0


def _load_alias_yaml() -> dict[str, dict | None]:
    """YAML 파일을 읽어 {입력명: {db_name, db_unit} | None} 형태로 반환."""
    if not _ALIAS_YAML_PATH.exists():
        archive("catalog.alias_yaml_missing", {"path": str(_ALIAS_YAML_PATH)})
        return {}

    try:
        import yaml
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


# Resolve

ResolveStatus = Literal[
    "matched",           # KAMIS alias/catalog/mapping 정확 매칭 → 실시간 도매가 조회 가능
    "recipe_matched",    # ingredient_recipe B2B 가격 매칭 (KAMIS 매핑 없음)
    "ambiguous",         # alias에서 명시적으로 자동 매칭 금지 + recipe에도 없음
    "not_in_catalog",    # alias 매핑은 있는데 어디에도 실제 데이터 없음
    "unmapped",          # 어디에도 없음 → Naver 폴백 대상
]


@dataclass
class ResolveResult:
    """재료명 해석 결과.

    status:
      - 'matched'         : KAMIS 카탈로그에 (db_name, db_unit)이 확인됨
                            (alias 직접매칭 OR mapping table 경유)
                            → Genie/direct_sql 1차 호출 대상
      - 'recipe_matched'  : KAMIS 매핑 없지만 ingredient_recipe에 B2B 유통가 존재
                            → Genie 건너뛰고 recipe 가격 직접 사용
      - 'ambiguous'       : alias 값이 None이고 recipe에도 없음 → Naver 폴백
      - 'not_in_catalog'  : alias 매핑은 있는데 KAMIS/recipe 모두 없음 → Naver 폴백
      - 'unmapped'        : 어디에도 없음 → Naver 폴백 대상
    """
    input_name: str
    status: ResolveStatus
    db_name: str | None
    db_unit: str | None
    reason: str
    recipe_info: dict | None = field(default=None)

    @property
    def has_catalog_target(self) -> bool:
        """'KAMIS 카탈로그에 있다고 확신할 수 있는' 케이스인지. direct_sql 대상 판정용."""
        return self.status == "matched"

    @property
    def has_recipe_price(self) -> bool:
        """recipe 테이블에서 B2B 가격을 가져올 수 있는지."""
        return self.status == "recipe_matched" and self.recipe_info is not None

    @property
    def has_any_price_source(self) -> bool:
        """DB 어딘가에서 가격을 가져올 수 있는지 (Naver 불필요)."""
        return self.status in ("matched", "recipe_matched")


def resolve_ingredient(name: str) -> ResolveResult:
    """입력 재료명을 카탈로그+alias+mapping+recipe로 해석.

    호출 순서 (4단계 탐색):
      1. alias에 키가 있는가?
         - 값이 dict: db_name이 KAMIS 카탈로그에 있으면 'matched'
         - 값이 None: 'ambiguous' → step 4로 (recipe에서 구제 가능)
      2. alias에 키가 없으면: KAMIS 카탈로그 자체에 정확명이 있는지 확인
         - 있으면 'matched'
      3. mapping 테이블에 이 재료가 있는가? (confidence ≥ 0.5)
         - 있으면 'matched' (kamis_name/kamis_unit으로 실시간 조회 가능)
      4. ingredient_recipe 카탈로그 확인
         - 있으면 'recipe_matched' (B2B 유통가 직접 사용)
         - 없으면 'unmapped' (Naver 폴백)
    """
    name = (name or "").strip()
    if not name:
        return ResolveResult(name, "unmapped", None, None, "empty_input")

    alias_table = get_alias_table()
    catalog = get_catalog()
    mapping_table = get_mapping_table()
    recipe_catalog = get_recipe_catalog()

    # ── 1. alias 우선 ──
    if name in alias_table:
        alias_val = alias_table[name]
        if alias_val is None:
            # ambiguous: recipe에서 구제 시도
            recipe_info = _lookup_recipe(name, recipe_catalog)
            if recipe_info:
                return ResolveResult(
                    name, "recipe_matched", None, None,
                    "alias_ambiguous_but_recipe_has_price",
                    recipe_info=recipe_info,
                )
            return ResolveResult(name, "ambiguous", None, None, "alias_explicit_ambiguous")

        db_name = alias_val["db_name"]
        db_unit = alias_val["db_unit"]
        catalog_units = catalog.get(db_name, set())
        if db_unit in catalog_units:
            return ResolveResult(name, "matched", db_name, db_unit, "alias_hit_catalog")
        # alias 매핑은 있지만 KAMIS에 해당 단위 없음 → recipe 체크
        recipe_info = _lookup_recipe(name, recipe_catalog)
        if recipe_info:
            return ResolveResult(
                name, "recipe_matched", db_name, db_unit,
                f"alias_maps_to_{db_name}/{db_unit}_not_in_catalog_but_recipe_has_price",
                recipe_info=recipe_info,
            )
        return ResolveResult(
            name, "not_in_catalog", db_name, db_unit,
            f"alias_maps_to_{db_name}/{db_unit}_but_catalog_has_{sorted(catalog_units) or 'none'}",
        )

    # ── 2. alias 미등록 — KAMIS 카탈로그 직접 매칭 ──
    if name in catalog:
        units = catalog[name]
        if units:
            unit = _best_unit(units)
            return ResolveResult(name, "matched", name, unit, "direct_catalog_hit")

    # 공백 제거 후 KAMIS 카탈로그 재시도
    normalized = _normalize_name(name)
    if normalized != name and normalized in catalog:
        units = catalog[normalized]
        if units:
            unit = _best_unit(units)
            return ResolveResult(name, "matched", normalized, unit, "normalized_catalog_hit")

    # 접두사 제거 후 KAMIS 카탈로그 재시도
    stripped = _strip_cooking_prefix(normalized)
    if stripped and stripped in catalog:
        units = catalog[stripped]
        if units:
            unit = _best_unit(units)
            return ResolveResult(name, "matched", stripped, unit, "prefix_stripped_catalog_hit")

    # ── 3. mapping 테이블로 KAMIS 매핑 확인 ──
    # recipe→KAMIS 매핑이 있는 재료는 matched로 승격 (실시간 도매가 조회 가능)
    mapping_entry = mapping_table.get(name) or mapping_table.get(normalized)
    if not mapping_entry and stripped:
        mapping_entry = mapping_table.get(stripped)
    if mapping_entry:
        kamis_name = mapping_entry["kamis_name"]
        kamis_unit = mapping_entry["kamis_unit"]
        # 실제로 KAMIS 카탈로그에 존재하는지 최종 확인
        if kamis_name in catalog and kamis_unit in catalog.get(kamis_name, set()):
            return ResolveResult(
                name, "matched", kamis_name, kamis_unit,
                f"mapping_table_hit_{mapping_entry['method']}_conf{mapping_entry['confidence']:.2f}",
            )

    # ── 4. KAMIS에 없음 — ingredient_recipe 폴백 ──
    recipe_info = _lookup_recipe(name, recipe_catalog)
    if recipe_info:
        return ResolveResult(
            name, "recipe_matched", None, None,
            "no_kamis_but_recipe_has_price",
            recipe_info=recipe_info,
        )

    return ResolveResult(name, "unmapped", None, None, "no_alias_no_catalog_no_mapping_no_recipe")


def resolve_many(names: list[str]) -> list[ResolveResult]:
    """여러 재료를 한 번에 해석. 중복은 첫 등장만 유지."""
    seen: set[str] = set()
    results: list[ResolveResult] = []
    for n in names:
        key = (n or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(resolve_ingredient(key))
    return results



def direct_sql_query_kamis(targets: list[tuple[str, str, str]]) -> dict:
    """KAMIS 테이블에 대해 (재료명, 단위) 조합으로 직접 SQL 조회.

    Genie 우회용. 최근 30일 평균 도매가를 반환.

    Args:
        targets: [(input_name, db_name, db_unit), ...]

    Returns:
        {
            "results": {input_name: {"price": int, "unit": str, "source": "kamis_direct"}},
            "found": [input_name, ...],
            "not_found": [input_name, ...],
            "error": Optional[str],
        }
    """
    if not targets:
        return {"results": {}, "found": [], "not_found": [], "error": None}

    where_clauses = []
    db_to_input: dict[tuple[str, str], str] = {}
    for input_name, db_name, db_unit in targets:
        safe_name = (db_name or "").replace("'", "''")
        safe_unit = (db_unit or "").replace("'", "''")
        where_clauses.append(f"(`재료명` = '{safe_name}' AND `단위` = '{safe_unit}')")
        db_to_input[(db_name, db_unit)] = input_name

    sql = f"""
SELECT `재료명`, `단위`, ROUND(AVG(`가격`)) AS `평균가격`, COUNT(*) AS `행수`
FROM silver.ingredient.ingredient
WHERE ({" OR ".join(where_clauses)})
  AND `날짜` >= DATE_SUB(CURRENT_DATE(), 30)
  AND `가격` IS NOT NULL
GROUP BY `재료명`, `단위`
""".strip()

    try:
        w = _get_ws_client()
        warehouse_id = _get_warehouse_id(w)
        if not warehouse_id:
            return {"results": {}, "found": [], "not_found": [t[0] for t in targets],
                    "error": "warehouse_not_found"}

        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="50s",
        )
        if not resp.status or resp.status.state != StatementState.SUCCEEDED:
            err_msg = resp.status.error.message if resp.status and resp.status.error else "unknown"
            return {"results": {}, "found": [], "not_found": [t[0] for t in targets],
                    "error": f"sql_failed: {err_msg}"}

        results: dict[str, dict] = {}
        found: list[str] = []
        for row in (resp.result.data_array or []):
            db_name = str(row[0] or "").strip()
            db_unit = str(row[1] or "").strip()
            try:
                avg_price = int(float(row[2])) if row[2] is not None else None
            except (TypeError, ValueError):
                avg_price = None
            if avg_price is None or avg_price <= 0:
                continue
            input_name = db_to_input.get((db_name, db_unit), db_name)
            results[input_name] = {
                "price": avg_price,
                "unit": db_unit,
                "db_name": db_name,
                "source": "kamis_direct",
            }
            found.append(input_name)

        not_found = [t[0] for t in targets if t[0] not in found]
        return {"results": results, "found": found, "not_found": not_found, "error": None}

    except Exception as e:
        return {"results": {}, "found": [], "not_found": [t[0] for t in targets],
                "error": f"exception: {str(e)}"}


def get_recipe_prices_for_items(names: list[str]) -> dict[str, dict]:
    """여러 재료의 recipe B2B 가격을 한 번에 조회.

    Returns:
        {input_name: {"price": int, "unit": str, "price_per_kg": int|None, ...}}
    """
    recipe_catalog = get_recipe_catalog()
    results: dict[str, dict] = {}

    for name in names:
        name = (name or "").strip()
        if not name:
            continue
        info = _lookup_recipe(name, recipe_catalog)
        if info and info.get("price"):
            ri = RecipeIngredientInfo(**info)
            results[name] = {
                "price": ri.price,
                "unit": ri.unit,
                "price_per_kg": ri.price_per_kg,
                "product_name": ri.product_name,
                "source": "recipe_b2b",
                "lv1": ri.lv1,
                "lv2": ri.lv2,
            }

    return results
