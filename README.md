# CineMatch — Movie Recommendation System

A content-based movie recommendation application built with **Python, Pandas, Scikit-learn, TF-IDF, cosine similarity, FastAPI and Streamlit**.

CineMatch recommends movies by comparing their genres, keywords, top-billed cast, director and plot overview. It includes a Streamlit user interface and a REST API.

## Features

- Content-based filtering using movie metadata.
- JSON parsing and feature engineering for genres, keywords, cast and director.
- TF-IDF text vectorization with unigrams and bigrams.
- Cosine similarity to rank similar movies.
- Partial title search and basic typo-tolerant title matching.
- Streamlit interactive interface.
- FastAPI endpoints with interactive Swagger documentation.
- Sparse TF-IDF matrix to reduce memory use compared with a dense similarity matrix.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, SciPy, Joblib, FastAPI, Uvicorn, Streamlit.

## Dataset

This project uses the **TMDB 5000 Movie Dataset** from Kaggle:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download and extract the archive. Put the following CSV files in `data/`:

```text
data/
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv
```

The movies file contains fields such as title, overview, genres and keywords. The credits file contains cast and crew information. The two files are joined using `id` and `movie_id`.

The dataset contains movie information collected from TMDB and is not a live catalogue. Check the dataset page for its terms and attribution requirements before redistributing the data.

## Project structure

```text
Movie-Recommendation-System/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   └── recommender.py
├── artifacts/                 # generated locally; ignored by Git
├── app.py                     # Streamlit frontend
├── train.py                   # data preparation and model training
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (Windows / Anaconda / terminal)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Create and activate an environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

If you use Anaconda instead:

```bash
conda create -n cinematch python=3.11
conda activate cinematch
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the Kaggle dataset and put both CSV files in the `data` folder. Check that the filenames match exactly.

### 5. Train the recommendation model

Run from the project root:

```bash
python train.py
```

The script preprocesses the data, creates TF-IDF vectors and saves:

```text
artifacts/movie_recommender.joblib
```

The `artifacts` folder is generated automatically. The model file is excluded from Git; each user should generate it by running the training script.

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

### 7. Run the FastAPI backend

Open a second terminal, activate the same environment, go to the project root and run:

```bash
uvicorn api.main:app --reload
```

API base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check API and loaded movie count |
| GET | `/movies/search?q=avatar` | Search movie titles |
| GET | `/movies/Avatar` | Get details for a movie |
| POST | `/recommend` | Get similar movie recommendations |

Example POST body:

```json
{
  "title": "Avatar",
  "n_recommendations": 5
}
```

Example using Python:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/recommend",
    json={"title": "Avatar", "n_recommendations": 5},
    timeout=30,
)
print(response.json())
```

## How the recommendation engine works

1. **Load and merge:** Read the two TMDB CSV files and merge movie metadata with credits.
2. **Parse metadata:** Extract names from JSON-formatted genres, keywords, cast and crew columns.
3. **Feature engineering:** Keep genres, keywords, top five cast members, director and overview. Combine these into a text field called `tags`.
4. **TF-IDF:** Convert the text into numerical vectors. Common English stop words are removed and one-/two-word terms are considered.
5. **Cosine similarity:** Compare the selected movie vector with all movie vectors and rank the most similar items.
6. **Serve predictions:** Streamlit provides the user interface; FastAPI exposes the same recommendation logic through REST endpoints.

This is a **content-based** recommender. It does not learn from user ratings or user-to-user behaviour, so it is different from collaborative filtering.

## Important notes

- The first training run can take a little time depending on your computer.
- The implementation uses a sparse TF-IDF matrix and calculates similarity for the selected movie when requested. It does not save a full dense movie-by-movie similarity matrix.
- Similarity is a text-content similarity score, not a probability that someone will like a movie.
- This project is an educational portfolio project, not an official TMDB product.
- Do not commit virtual environments, raw datasets, API secrets or large generated artifacts.

## Future improvements

- Add TMDB API integration for live posters and current movie details.
- Add genre, language and release-year filters.
- Add user feedback and collaborative filtering.
- Add automated tests, Docker and cloud deployment.
- Add recommendation evaluation using a labelled or user-feedback dataset.

## Author

**Kunal Bhardwaj**  
Aspiring Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/kunal-bhardwaj-3bba84372) · [GitHub](https://github.com/Kunalkunal-bhardwaj-ml)

## Acknowledgements

- Dataset: TMDB 5000 Movie Dataset on Kaggle.
- Libraries: Scikit-learn, Pandas, FastAPI and Streamlit.
- Movie data is provided by TMDB dataset contributors. This project is not endorsed or certified by TMDB.
