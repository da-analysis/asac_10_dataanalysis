import time
from backend.nodes.price_search import price_search_node
import mlflow
from concurrent.futures import ThreadPoolExecutor
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient


# 테스트할 재료 리스트 (최소 15~20개 이상 넉넉하게 준비)
test_ingredients = [
    "돼지고기", "소고기", "닭고기", "두부", "김치",
    "양파", "대파", "감자", "당근", "애호박",
    "버섯", "고추", "마늘", "계란", "우유"
]

# 실제 챗봇의 state와 유사한 가짜 상태(dummy state) 생성
dummy_state = {
    "entities": {"ingredient": test_ingredients},
    "recipe_info": {},
    "messages": []
}

print(f"🚀 가격 검색 노드 테스트 시작 (재료 {len(test_ingredients)}개)...")
start_time = time.time()

# 노드 직접 실행
result = price_search_node(dummy_state)

end_time = time.time()

# 결과 출력
print("-" * 50)
print(f"⏱️ 소요 시간: {end_time - start_time:.2f}초")

price_info = result.get("price_info", {})
unavailable = price_info.get("unavailable", [])

print(f"✅ 검색 실패(unavailable) 개수: {len(unavailable)}개")
print(f"   내역: {unavailable}")
if "error_log" in result:
    print(f"⚠️ 에러 발생: {result['error_log']}")
