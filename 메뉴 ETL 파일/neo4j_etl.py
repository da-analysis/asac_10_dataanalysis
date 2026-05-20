# Databricks notebook source
# MAGIC %pip install neo4j --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Neo4j 연결

# COMMAND ----------

from neo4j import GraphDatabase

NEO4J_URI = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_uri")
NEO4J_USERNAME = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_username")
NEO4J_PASSWORD = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# 연결 테스트
with driver.session() as session:
    result = session.run("RETURN 'Connected!' AS msg")
    print(result.single()["msg"])

print("Neo4j Aura 연결")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 기존 데이터 삭제 (필요시)

# COMMAND ----------

# 기존 데이터를 전부 삭제합니다
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("기존 데이터 삭제 완료")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 샘플 데이터 준비
# MAGIC
# MAGIC - 레시피: `silver.10000recipe.recipes`
# MAGIC - 재료: **`silver.10000recipe.ingredients_final`** (51번 최종본)

# COMMAND ----------

# 노드수 20만/관계수 40만 제한 고려 — 레시피 5만개 샘플
SAMPLE_SIZE = 45000

recipes_df = spark.sql(f"""
    SELECT * FROM silver.`10000recipe`.recipes
    WHERE CKG_NM IS NOT NULL
    LIMIT {SAMPLE_SIZE}
""")

recipe_ids = [row.RCP_SNO for row in recipes_df.select("RCP_SNO").collect()]
recipe_ids_str = ",".join([str(x) for x in recipe_ids])

# v2: ingredients_final 사용 — 깔끔한 core_name + quantity_text만
# 같은 (RCP_SNO, core_name)이 여러 row면 첫번째 quantity만 사용 (대표)
ingredients_df = spark.sql(f"""
    SELECT
        RCP_SNO,
        core_name,
        FIRST(lv1) AS lv1,
        FIRST(lv2) AS lv2,
        FIRST(NULLIF(quantity_text, '')) AS quantity_text
    FROM silver.`10000recipe`.ingredients_final
    WHERE RCP_SNO IN ({recipe_ids_str})
      AND core_name IS NOT NULL
      AND core_name != ''
    GROUP BY RCP_SNO, core_name
""")

unique_ingredients = ingredients_df.select("core_name", "lv1", "lv2").dropDuplicates(["core_name"])

print(f"레시피 => {recipes_df.count()}개")
print(f"재료 매핑 => {ingredients_df.count()}개")
print(f"고유 재료 => {unique_ingredients.count()}개")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 인덱스/제약조건 생성

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
# MAGIC ## 5. Recipe 노드 생성 (메타데이터 포함)

# COMMAND ----------

recipes_list = [row.asDict() for row in recipes_df.collect()]

for i in range(0, len(recipes_list), 500):
    batch = recipes_list[i:i+500]
    with driver.session() as session:
        session.run("""
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
        """, items=batch)
    print(f"  Recipe: {min(i+500, len(recipes_list))}/{len(recipes_list)}")

print(f"Recipe 노드 {len(recipes_list)}개 생성")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ingredient 노드 생성

# COMMAND ----------

ing_list = [row.asDict() for row in unique_ingredients.collect()]

with driver.session() as session:
    session.run("""
        UNWIND $items AS item
        MERGE (i:Ingredient {name: item.core_name})
        SET i.lv1 = item.lv1,
            i.lv2 = item.lv2
    """, items=ing_list)

print(f"Ingredient 노드 {len(ing_list)}개 생성")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CONTAINS 관계 생성 (수량/조리상태 메타 포함)
# MAGIC
# MAGIC 관계 속성:
# MAGIC - `quantity` (legacy, 기존 호환용 → quantity_text와 동일)
# MAGIC - `quantity_text`: 사용자가 본 원본 표시 (예: "1큰술", "1/2개")
# MAGIC - `quantity_g`: g 환산값 (원가 계산용)
# MAGIC - `quantity_count`, `quantity_unit`: 개수 단위 (예: 1.0 / "개")
# MAGIC - `cooking_state`: 다진/구운/채썬 등

# COMMAND ----------

rel_list = [row.asDict() for row in ingredients_df.collect()]

for i in range(0, len(rel_list), 2000):
    batch = rel_list[i:i+2000]
    with driver.session() as session:
        session.run("""
            UNWIND $items AS item
            MATCH (r:Recipe {rcp_sno: item.RCP_SNO})
            MATCH (i:Ingredient {name: item.core_name})
            MERGE (r)-[c:CONTAINS]->(i)
            SET c.quantity = item.quantity_text
        """, items=batch)
    print(f"  CONTAINS: {min(i+2000, len(rel_list))}/{len(rel_list)}")

print(f"CONTAINS 관계 {len(rel_list)}개 생성")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 검증

# COMMAND ----------

queries = [
    ("Recipe", "MATCH (r:Recipe) RETURN count(r) AS count"),
    ("Ingredient", "MATCH (i:Ingredient) RETURN count(i) AS count"),
    ("CONTAINS 관계", "MATCH ()-[c:CONTAINS]->() RETURN count(c) AS count"),
]

print("[메뉴 그래프 통계]")
with driver.session() as session:
    for label, query in queries:
        result = session.run(query)
        count = result.single()["count"]
        print(f"  {label:20s}: {count:>8,}개")

print("\n[샘플 (김치찌개로 샘플)]")
with driver.session() as session:
    result = session.run("""
        MATCH (r:Recipe)-[c:CONTAINS]->(i:Ingredient)
        WHERE r.name CONTAINS '김치찌개'
        RETURN r.name AS name, i.name AS ing, c.quantity AS qty
        LIMIT 10
    """)
    for record in result:
        print(f"  {record['name']} → {record['ing']} ({record['qty'] or '수량없음'})")

# COMMAND ----------

######### + 조리단계 누락으로 속성으로 추가 #############

# COMMAND ----------

# 조리단계를 레시피별로 합치기
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

print(f"조리단계 있는 레시피: {steps_agg.count()}개")
steps_agg.show(3, truncate=80)

# COMMAND ----------

steps_list = [row.asDict() for row in steps_agg.collect()]

for i in range(0, len(steps_list), 500):
    batch = steps_list[i:i+500]
    with driver.session() as session:
        session.run("""
            UNWIND $items AS item
            MATCH (r:Recipe {rcp_sno: item.rcp_sno})
            SET r.steps = item.steps
        """, items=batch)
    print(f"  Steps: {min(i+500, len(steps_list))}/{len(steps_list)}")

print(f"조리단계 {len(steps_list)}개 레시피에 추가 완료")

# COMMAND ----------

with driver.session() as session:
    result = session.run("""
        MATCH (r:Recipe)
        WHERE r.name = '김치찌개' AND r.steps IS NOT NULL
        RETURN r.name, r.steps
        LIMIT 1
    """)
    record = result.single()
    if record:
        print(f"레시피: {record['r.name']}")
        print(f"조리단계:\n{record['r.steps']}")
    else:
        print("조리단계 없음")
