from pathlib import Path
import argparse

from src.recommender import train_recommender

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train the TMDB movie recommender.")
    parser.add_argument("--movies", default=ROOT / "data" / "tmdb_5000_movies.csv")
    parser.add_argument("--credits", default=ROOT / "data" / "tmdb_5000_credits.csv")
    parser.add_argument("--output", default=ROOT / "artifacts" / "movie_recommender.joblib")
    args = parser.parse_args()

    result = train_recommender(args.movies, args.credits, args.output)
    print("\nTraining completed successfully.")
    print(f"Movies processed : {result['movie_count']}")
    print(f"TF-IDF features  : {result['feature_count']}")
    print(f"Saved artifact   : {result['artifact_path']}")


if __name__ == "__main__":
    main()
