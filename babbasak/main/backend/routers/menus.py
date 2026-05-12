from fastapi import APIRouter
from pydantic import BaseModel
from backend.db import get_menus_by_ingredient

router = APIRouter(tags=["menus"])


class RecommendRequest(BaseModel):
    ingredients: list[str]
    target_margin: float = 60.0


@router.post("/recommend")
def recommend_menus(req: RecommendRequest):
    results = []
    for ingredient in req.ingredients:
        results.extend(get_menus_by_ingredient(ingredient))

    seen = set()
    unique = []
    for m in sorted(results, key=lambda x: x.get("margin", 0), reverse=True):
        if m["name"] not in seen:
            seen.add(m["name"])
            unique.append(m)

    return unique[:10]
