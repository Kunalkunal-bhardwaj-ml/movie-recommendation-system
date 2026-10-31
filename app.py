from pathlib import Path

import streamlit as st

from src.recommender import MovieRecommender

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "movie_recommender.joblib"

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 CineMatch")
st.subheader("Discover movies similar to the ones you love.")
st.write(
    "A content-based recommendation system that compares movie genres, "
    "keywords, cast, director and plot overview."
)


@st.cache_resource
def load_recommender():
    return MovieRecommender(ARTIFACT_PATH)


try:
    recommender = load_recommender()
except FileNotFoundError:
    st.error("Trained model not found. First download the dataset and run `python train.py`.")
    st.stop()

st.caption(f"Recommendations available from {len(recommender.movies):,} movies.")

with st.form("recommendation_form"):
    movie_titles = sorted(recommender._titles, key=str.casefold)
    selected_movie = st.selectbox("Choose a movie", movie_titles)
    number = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    submitted = st.form_submit_button("Recommend movies", type="primary")

if submitted:
    try:
        movie = recommender.movie_details(selected_movie)
        recommendations = recommender.recommend(selected_movie, number)

        st.markdown("---")
        st.subheader(f"Because you selected: {movie['title']}")
        if movie["overview"]:
            st.write(movie["overview"])
        if movie["genres"]:
            st.caption("Genres: " + ", ".join(movie["genres"]))

        st.subheader("You may also like")
        columns = st.columns(2)
        for index, item in enumerate(recommendations):
            with columns[index % 2]:
                with st.container(border=True):
                    st.markdown(f"### {index + 1}. {item['title']}")
                    if item["release_date"]:
                        st.caption(f"Release date: {item['release_date']}")
                    if item["genres"]:
                        st.write("**Genres:** " + ", ".join(item["genres"]))
                    if item["director"]:
                        st.write(f"**Director:** {item['director']}")
                    if item["vote_average"] is not None:
                        st.write(f"**TMDB rating:** {item['vote_average']}/10")
                    st.progress(min(max(item["similarity"], 0.0), 1.0))
                    st.caption(f"Content similarity: {item['similarity']:.1%}")
                    if item["overview"]:
                        with st.expander("Plot overview"):
                            st.write(item["overview"])
    except ValueError as exc:
        st.warning(str(exc))

st.markdown("---")
st.caption("Built with Python, Pandas, Scikit-learn, FastAPI and Streamlit.")
