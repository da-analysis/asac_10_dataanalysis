# Databricks notebook source
# MAGIC %pip install neo4j --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Neo4j 연결
# MAGIC ⚠️ neo4j-scope/neo4j_uri 가 **새 인스턴스(9165c00a)** 를 가리키는지 먼저 확인!

# COMMAND ----------

from neo4j import GraphDatabase

NEO4J_URI = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_uri")
NEO4J_USERNAME = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_username")
NEO4J_PASSWORD = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

with driver.session() as session:
    msg = session.run("RETURN 'Connected!' AS msg").single()["msg"]
    # 적재 대상이 새 인스턴스인지 눈으로 확인
    print(f"{msg}  →  {NEO4J_URI}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-1. 스트리밍 배치 헬퍼
# MAGIC 전체 적재는 `.collect()`로 드라이버에 다 못 올림(220만 관계 → OOM).
# MAGIC `toLocalIterator()`로 파티션 단위로 흘려보내며 배치 단위로만 메모리에 유지.

# COMMAND ----------

def iter_batches(df, size):
    """Spark df를 드라이버 전체 적재 없이 size개씩 dict 배치로 yield."""
    batch = []
    for row in df.toLocalIterator():
        batch.append(row.asDict())
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_batches(df, size, cypher, label, total=None):
    """df를 배치로 끊어 cypher(UNWIND $items)를 실행. 단일 세션 재사용."""
    done = 0
    with driver.session() as session:
        for batch in iter_batches(df, size):
            session.run(cypher, items=batch)
            done += len(batch)
            tail = f"/{total:,}" if total else ""
            print(f"  {label}: {done:,}{tail}")
    print(f"{label} 완료: {done:,}")
    return done

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 기존 데이터 삭제
# MAGIC 전체 재적재이므로 깨끗하게 비우고 시작. (관계 많으면 시간 걸림)

# COMMAND ----------

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
print("기존 데이터 삭제 완료")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 데이터 준비 (전체 — LIMIT/IN 필터 없음)
# MAGIC - 레시피: `silver.10000recipe.recipes` 전체
# MAGIC - 재료: `silver.10000recipe.ingredients_final` 전체
# MAGIC - ★ 기존의 `WHERE RCP_SNO IN (...)` 제거 — 전체 적재라 불필요하고 대량이면 SQL이 터짐

# COMMAND ----------

recipes_df = spark.sql("""
    SELECT * FROM silver.`10000recipe`.recipes
    WHERE CKG_NM IS NOT NULL
""")

# 같은 (RCP_SNO, core_name) 여러 row면 대표 수량 1개만
ingredients_df = spark.sql("""
    SELECT
        RCP_SNO,
        core_name,
        FIRST(lv1) AS lv1,
        FIRST(lv2) AS lv2,
        FIRST(NULLIF(quantity_text, '')) AS quantity_text
    FROM silver.`10000recipe`.ingredients_final
    WHERE core_name IS NOT NULL AND core_name != ''
    GROUP BY RCP_SNO, core_name
""")

unique_ingredients = ingredients_df.select("core_name", "lv1", "lv2").dropDuplicates(["core_name"])

# count는 한 번만 (action) — 이후엔 재사용 위해 cache 대신 변수 보관
# ※ serverless라 .cache()/.persist() 금지 → count만 뽑고 df는 그대로 재질의
n_recipes = recipes_df.count()
n_rels = ingredients_df.count()
n_ings = unique_ingredients.count()
print(f"레시피      => {n_recipes:,}개")
print(f"재료 매핑   => {n_rels:,}개")
print(f"고유 재료   => {n_ings:,}개")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 인덱스/제약조건 (적재 前 필수)
# MAGIC MERGE가 인덱스를 타야 함. 없으면 노드 늘수록 풀스캔 → 수십 배 느려짐.

# COMMAND ----------

constraints = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Recipe) REQUIRE r.rcp_sno IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (r:Recipe) ON (r.name)",
    "CREATE INDEX IF NOT EXISTS FOR (r:Recipe) ON (r.category)",
    "CREATE INDEX IF NOT EXISTS FOR (i:Ingredient) ON (i.lv1)",
]
with driver.session() as session:
    for c in constraints:
        session.run(c)
print("인덱스/제약조건 생성")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Recipe 노드 (스트리밍 배치)

# COMMAND ----------

RECIPE_CYPHER = """
    UNWIND $items AS item
    MERGE (r:Recipe {rcp_sno: item.RCP_SNO})
    SET r.name = item.CKG_NM,
        r.title = item.RCP_TTL,
        r.cooking_method = item.CKG_MTH_ACTO_NM,
        r.situation = item.CKG_STA_ACTO_NM,
        r.main_ingredient = item.CKG_MTRL_ACTO_NM,
        r.kind = item.CKG_KND_ACTO_NM,
        r.servings = item.CKG_INBUN_NM,
        r.difficulty = item.CKG_DODF_NM,
        r.cooking_time = item.CKG_TIME_NM,
        r.view_count = item.INQ_CNT,
        r.recommend_count = item.RCMM_CNT,
        r.scrap_count = item.SRAP_CNT,
        r.description = item.CKG_IPDC,
        r.image_url = item.RCP_IMG_URL
"""
run_batches(recipes_df, 1000, RECIPE_CYPHER, "Recipe", total=n_recipes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ingredient 노드 (고유 재료 — 작아서 한 번에)

# COMMAND ----------

ing_list = [row.asDict() for row in unique_ingredients.collect()]  # 5만 내외, 안전
with driver.session() as session:
    session.run("""
        UNWIND $items AS item
        MERGE (i:Ingredient {name: item.core_name})
        SET i.lv1 = item.lv1, i.lv2 = item.lv2
    """, items=ing_list)
print(f"Ingredient 노드 {len(ing_list):,}개 생성")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CONTAINS 관계 (스트리밍 배치 — 가장 큼)

# COMMAND ----------

CONTAINS_CYPHER = """
    UNWIND $items AS item
    MATCH (r:Recipe {rcp_sno: item.RCP_SNO})
    MATCH (i:Ingredient {name: item.core_name})
    MERGE (r)-[c:CONTAINS]->(i)
    SET c.quantity = item.quantity_text
"""
run_batches(ingredients_df, 5000, CONTAINS_CYPHER, "CONTAINS", total=n_rels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 조리단계 속성 추가 (스트리밍 배치)

# COMMAND ----------

steps_agg = spark.sql("""
    SELECT CAST(RCP_SNO AS INT) AS rcp_sno,
           concat_ws('\n', collect_list(
               concat(COOKING_NO, '. ', COALESCE(COOKING_DC_CLEAN, ''))
           )) AS steps
    FROM silver.`10000recipe`.cooking_steps
    WHERE is_noise = false
      AND COOKING_DC_CLEAN IS NOT NULL AND COOKING_DC_CLEAN != ''
    GROUP BY RCP_SNO
""")
n_steps = steps_agg.count()
print(f"조리단계 있는 레시피: {n_steps:,}개")

STEPS_CYPHER = """
    UNWIND $items AS item
    MATCH (r:Recipe {rcp_sno: item.rcp_sno})
    SET r.steps = item.steps
"""
run_batches(steps_agg, 1000, STEPS_CYPHER, "Steps", total=n_steps)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. 검증

# COMMAND ----------

queries = [
    ("Recipe", "MATCH (r:Recipe) RETURN count(r) AS count"),
    ("Ingredient", "MATCH (i:Ingredient) RETURN count(i) AS count"),
    ("CONTAINS 관계", "MATCH ()-[c:CONTAINS]->() RETURN count(c) AS count"),
    ("steps 보유 Recipe", "MATCH (r:Recipe) WHERE r.steps IS NOT NULL RETURN count(r) AS count"),
]
print("[메뉴 그래프 통계]")
with driver.session() as session:
    for label, q in queries:
        print(f"  {label:20s}: {session.run(q).single()['count']:>10,}개")

print("\n[샘플 — 김치찌개]")
with driver.session() as session:
    for rec in session.run("""
        MATCH (r:Recipe)-[c:CONTAINS]->(i:Ingredient)
        WHERE r.name CONTAINS '김치찌개'
        RETURN r.name AS name, i.name AS ing, c.quantity AS qty LIMIT 10
    """):
        print(f"  {rec['name']} → {rec['ing']} ({rec['qty'] or '수량없음'})")

# COMMAND ----------

driver.close()
print("드라이버 종료")

