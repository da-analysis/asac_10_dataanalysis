import backend.databricks_db as databricks_db
from fastapi import FastAPI
from backend.routers import chatbot as chatbot_router
import mlflow

# mlflow 설정. 
mlflow.langchain.autolog()
mlflow.set_experiment('/Shared/babasak_tracing')


app = FastAPI()
app.include_router(chatbot_router.router, prefix="/api/chatbot")


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
