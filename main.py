from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.recommender import MovieRecommender

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "movie_recommender.joblib"

app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendations using TMDB metadata, TF-IDF and cosine similarity.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    return MovieRecommender(ARTIFACT_PATH)


class RecommendationRequest(BaseModel):
    title: str = Field(..., min_length=1, examples=["Avatar"])
    n_recommendations: int = Field(default=5, ge=1, le=20)


@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    try:
        recommender = get_recommender()
        return {"status": "ok", "movies_loaded": len(recommender.movies)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    try:
        return {"query": q, "results": get_recommender().search_titles(q, limit)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/movies/{title}")
def movie_details(title: str):
    try:
        return get_recommender().movie_details(title)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    try:
        return {
            "input_movie": request.title,
            "recommendations": get_recommender().recommend(
                request.title, request.n_recommendations
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
