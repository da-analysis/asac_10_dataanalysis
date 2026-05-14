from fastapi import APIRouter
from backend import databricks_db

router = APIRouter(tags=["ingredients"])


@router.get("/")
def list_ingredients():
    try:
        live = databricks_db.get_ingredient_prices_live()
        if live:
            return {"items": live, "source": "databricks"}
        return {"items": [], "source": "none", "error": databricks_db.last_error}
    except Exception as exc:
        return {"items": [], "source": "none", "error": str(exc)}
