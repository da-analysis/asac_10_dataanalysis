import os
from neo4j import GraphDatabase

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def get_all_ingredients() -> list[dict]:
    with get_driver().session() as session:
        result = session.run("""
            MATCH (i:Ingredient)
            RETURN i.name AS name, i.icon AS icon, i.price AS price, i.change AS change
        """)
        return [r.data() for r in result]


def get_ingredient_prices(name: str) -> list[dict]:
    with get_driver().session() as session:
        result = session.run("""
            MATCH (i:Ingredient {name: $name})-[:HAS_PRICE]->(p:Price)
            RETURN p.date AS date, p.value AS value
            ORDER BY p.date
        """, name=name)
        return [r.data() for r in result]


def get_menus_by_ingredient(name: str) -> list[dict]:
    with get_driver().session() as session:
        result = session.run("""
            MATCH (i:Ingredient {name: $name})<-[:USES]-(m:Menu)
            RETURN m.name AS name, m.margin AS margin, m.cost AS cost
            ORDER BY m.margin DESC
        """, name=name)
        return [r.data() for r in result]


def get_chatbot_context(keyword: str) -> list[dict]:
    with get_driver().session() as session:
        result = session.run("""
            MATCH (m:Menu)-[:USES]->(i:Ingredient)
            WHERE i.name CONTAINS $keyword
            RETURN m.name AS menu, m.margin AS margin, collect(i.name) AS ingredients
            ORDER BY m.margin DESC
            LIMIT 5
        """, keyword=keyword[:10])
        return [r.data() for r in result]
