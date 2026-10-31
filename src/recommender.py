from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _parse_json_list(value: Any) -> list[dict]:
    """Safely parse TMDB's JSON-like columns."""
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []


def _names(value: Any, limit: int | None = None) -> list[str]:
    items = _parse_json_list(value)
    result = [str(item.get("name", "")).strip() for item in items if item.get("name")]
    return result[:limit] if limit else result


def _director(value: Any) -> str:
    for person in _parse_json_list(value):
        if person.get("job") == "Director" and person.get("name"):
            return str(person["name"]).strip()
    return ""


def _normalise_token(text: str) -> str:
    # Join multi-word names so TF-IDF treats them as a single feature.
    return re.sub(r"\s+", "", str(text).lower().strip())


def load_and_prepare_data(movies_path: str | Path, credits_path: str | Path) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    required_movies = {"id", "title", "genres", "keywords", "overview"}
    required_credits = {"movie_id", "cast", "crew"}
    missing_movies = required_movies - set(movies.columns)
    missing_credits = required_credits - set(credits.columns)
    if missing_movies or missing_credits:
        raise ValueError(
            f"Unexpected dataset columns. Missing from movies: {sorted(missing_movies)}; "
            f"missing from credits: {sorted(missing_credits)}"
        )

    credits = credits[["movie_id", "cast", "crew"]].copy()
    merged = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    merged = merged.drop_duplicates(subset=["id"]).copy()
    merged["title"] = merged["title"].fillna("").astype(str).str.strip()
    merged["overview"] = merged["overview"].fillna("").astype(str)
    merged = merged[merged["title"] != ""].copy()

    merged["genres_list"] = merged["genres"].apply(_names)
    merged["keywords_list"] = merged["keywords"].apply(_names)
    merged["cast_list"] = merged["cast"].apply(lambda x: _names(x, limit=5))
    merged["director"] = merged["crew"].apply(_director)

    def build_tags(row: pd.Series) -> str:
        parts = (
            row["genres_list"]
            + row["keywords_list"]
            + row["cast_list"]
            + ([row["director"]] if row["director"] else [])
            + [row["overview"]]
        )
        return " ".join(_normalise_token(part) for part in parts if part)

    merged["tags"] = merged.apply(build_tags, axis=1)
    merged = merged[merged["tags"].str.strip() != ""].copy()

    keep = ["id", "title", "overview", "genres_list", "cast_list", "director",
            "release_date", "vote_average", "vote_count", "popularity", "tags"]
    for col in ["release_date", "vote_average", "vote_count", "popularity"]:
        if col not in merged:
            merged[col] = None
    return merged[keep].reset_index(drop=True)


def train_recommender(movies_path: str | Path, credits_path: str | Path,
                      artifact_path: str | Path, max_features: int = 20000) -> dict:
    """Train TF-IDF content vectors and save a compact sparse artifact."""
    movies = load_and_prepare_data(movies_path, credits_path)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(movies["tags"])
    artifact = {
        "movies": movies.drop(columns=["tags"]),
        "vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "version": 1,
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    return {
        "movie_count": len(movies),
        "feature_count": matrix.shape[1],
        "artifact_path": str(artifact_path),
    }


class MovieRecommender:
    def __init__(self, artifact_path: str | Path):
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}. Run `python train.py` first."
            )
        artifact = joblib.load(artifact_path)
        self.movies: pd.DataFrame = artifact["movies"].reset_index(drop=True)
        self.vectorizer: TfidfVectorizer = artifact["vectorizer"]
        self.matrix = artifact["tfidf_matrix"]
        self._titles = self.movies["title"].astype(str).tolist()
        self._title_lookup = {title.casefold(): i for i, title in enumerate(self._titles)}

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        query = query.strip().casefold()
        if not query:
            return []
        matches = [title for title in self._titles if query in title.casefold()]
        return matches[:max(1, min(limit, 50))]

    def _find_index(self, title: str) -> int:
        query = title.strip().casefold()
        if query in self._title_lookup:
            return self._title_lookup[query]
        from difflib import get_close_matches
        matches = get_close_matches(query, list(self._title_lookup), n=1, cutoff=0.55)
        if matches:
            return self._title_lookup[matches[0]]
        suggestions = self.search_titles(title, limit=5)
        hint = f" Similar titles: {', '.join(suggestions)}." if suggestions else ""
        raise ValueError(f"Movie '{title}' was not found in the dataset.{hint}")

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value

    def recommend(self, title: str, n_recommendations: int = 5) -> list[dict]:
        if not 1 <= n_recommendations <= 20:
            raise ValueError("n_recommendations must be between 1 and 20.")
        index = self._find_index(title)
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        results = []
        for candidate in ranked:
            if candidate == index:
                continue
            row = self.movies.iloc[int(candidate)]
            results.append({
                "id": int(row["id"]),
                "title": str(row["title"]),
                "overview": str(row["overview"]) if row["overview"] else "",
                "genres": row["genres_list"] if isinstance(row["genres_list"], list) else [],
                "director": row["director"] or "",
                "release_date": self._safe_value(row["release_date"]),
                "vote_average": self._safe_value(row["vote_average"]),
                "similarity": round(float(scores[candidate]), 4),
            })
            if len(results) == n_recommendations:
                break
        return results

    def movie_details(self, title: str) -> dict:
        index = self._find_index(title)
        row = self.movies.iloc[index]
        return {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "overview": str(row["overview"]) if row["overview"] else "",
            "genres": row["genres_list"] if isinstance(row["genres_list"], list) else [],
            "cast": row["cast_list"] if isinstance(row["cast_list"], list) else [],
            "director": row["director"] or "",
            "release_date": self._safe_value(row["release_date"]),
            "vote_average": self._safe_value(row["vote_average"]),
        }
