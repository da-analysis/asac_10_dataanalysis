# Databricks notebook source
# MAGIC %md
# MAGIC # Neo4j 메뉴 데이터 ETL
# MAGIC Databricks → Neo4j Aura로 메뉴/재료 데이터 로드

# COMMAND ----------

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
# with driver.session() as session:
#     session.run("MATCH (n) DETACH DELETE n")
# print("기존 데이터 삭제 완료")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 샘플 데이터 준비

# COMMAND ----------

# 레시피 샘플링 레시피 메뉴 5만개 넣을시, 관계 약 40만개로 노드 8만, 관계 40
# 이유 => 노드수 20만개 관계수 40만개 제한
SAMPLE_SIZE = 50000

recipes_df = spark.sql(f"""
    SELECT * FROM silver.`10000recipe`.recipes 
    WHERE CKG_NM IS NOT NULL 
    LIMIT {SAMPLE_SIZE}
""")

recipe_ids = [row.RCP_SNO for row in recipes_df.select("RCP_SNO").collect()]
recipe_ids_str = ",".join([str(x) for x in recipe_ids])

ingredients_df = spark.sql(f"""
    SELECT i.*, m.lv1, m.lv2, m.freq
    FROM silver.`10000recipe`.ingredients i
    LEFT JOIN silver.`10000recipe`.ingredient_master m
    ON i.canonical_name = m.std_name
    WHERE i.RCP_SNO IN ({recipe_ids_str})
""")

unique_ingredients = ingredients_df.select("canonical_name", "lv1", "lv2").dropDuplicates(["canonical_name"])

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
        MERGE (i:Ingredient {name: item.canonical_name})
        SET i.lv1 = item.lv1,
            i.lv2 = item.lv2
    """, items=ing_list)

print(f"Ingredient 노드 {len(ing_list)}개 생성")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CONTAINS 관계 생성

# COMMAND ----------

rel_list = [row.asDict() for row in ingredients_df.select("RCP_SNO", "canonical_name", "quantity").collect()]

for i in range(0, len(rel_list), 2000):
    batch = rel_list[i:i+2000]
    with driver.session() as session:
        session.run("""
            UNWIND $items AS item
            MATCH (r:Recipe {rcp_sno: item.RCP_SNO})
            MATCH (i:Ingredient {name: item.canonical_name})
            MERGE (r)-[c:CONTAINS]->(i)
            SET c.quantity = item.quantity
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

print("[\n샘플(김치찌개로 샘플)]")
with driver.session() as session:
    result = session.run("""
        MATCH (r:Recipe)-[c:CONTAINS]->(i:Ingredient)
        WHERE r.name CONTAINS '김치찌개'
        RETURN r.name, i.name, c.quantity
        LIMIT 10
    """)
    for record in result:
        print(f"  {record['r.name']} → {record['i.name']} ({record['c.quantity']})")
