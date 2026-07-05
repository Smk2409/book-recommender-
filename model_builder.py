"""
model_builder.py
-----------------
Builds the two recommendation models used by the Flask app and saves them
to /models as pickle files:

1. Popularity-Based Recommender
   Ranks all books using a weighted rating formula (the same idea IMDB uses
   for its Top 250) so books with few ratings don't unfairly outrank books
   with thousands of ratings just because their average is slightly higher.

2. Item-Based Collaborative Filtering Recommender
   Builds a user-book rating matrix (restricted to active users and popular
   books to keep the matrix a manageable size) and computes cosine
   similarity between books based on how users rated them. Books rated
   similarly by the same users end up "close" to each other.

Run this once before starting the Flask app:
    python model_builder.py
"""

import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
MODELS_DIR = "models"

# Minimum number of ratings a user must have made / a book must have
# received to be included in the collaborative-filtering matrix.
MIN_RATINGS_PER_USER = 150
MIN_RATINGS_PER_BOOK = 800


def load_data():
    print("Loading books.csv and ratings.csv ...")
    books = pd.read_csv(f"{DATA_DIR}/books.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

    # Keep only the columns we actually need from books.csv
    books = books[
        [
            "book_id",
            "title",
            "authors",
            "original_publication_year",
            "average_rating",
            "ratings_count",
            "image_url",
            "small_image_url",
        ]
    ].copy()

    print(f"  books:   {books.shape[0]:,} rows")
    print(f"  ratings: {ratings.shape[0]:,} rows")
    return books, ratings


def build_popularity_model(books: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted rating formula (IMDB-style):
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where
        R = book's own average rating
        v = number of ratings the book has
        m = minimum ratings required to be considered (we use the 75th
            percentile of ratings_count so only fairly well-rated books
            qualify for the "Top Picks" list)
        C = mean rating across the whole catalogue
    """
    print("Building popularity-based model ...")
    C = books["average_rating"].mean()
    m = books["ratings_count"].quantile(0.75)

    qualified = books[books["ratings_count"] >= m].copy()

    def weighted_rating(row, m=m, C=C):
        v = row["ratings_count"]
        R = row["average_rating"]
        return (v / (v + m) * R) + (m / (v + m) * C)

    qualified["weighted_score"] = qualified.apply(weighted_rating, axis=1)
    popular_df = qualified.sort_values("weighted_score", ascending=False).head(50)
    popular_df = popular_df.reset_index(drop=True)
    return popular_df


def build_collaborative_model(books: pd.DataFrame, ratings: pd.DataFrame):
    """
    Item-based collaborative filtering using cosine similarity over a
    user x book pivot table of ratings.
    """
    print("Building collaborative-filtering model ...")

    # Keep only active users and reasonably popular books, otherwise the
    # pivot table becomes huge and mostly empty (sparse) which slows
    # everything down without helping recommendation quality.
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index

    book_counts = ratings["book_id"].value_counts()
    popular_books = book_counts[book_counts >= MIN_RATINGS_PER_BOOK].index

    filtered_ratings = ratings[
        ratings["user_id"].isin(active_users) & ratings["book_id"].isin(popular_books)
    ]
    print(f"  filtered ratings used for similarity: {filtered_ratings.shape[0]:,} rows")

    merged = filtered_ratings.merge(books[["book_id", "title"]], on="book_id")

    # Some books share a title but not a book_id (rare) -- keep first
    # occurrence per (user, title) to avoid duplicate index errors in pivot.
    merged = merged.drop_duplicates(subset=["user_id", "title"])

    pt = merged.pivot_table(index="title", columns="user_id", values="rating")
    pt = pt.fillna(0).astype("float32")
    print(f"  pivot table shape: {pt.shape} (books x users)")

    similarity_scores = cosine_similarity(pt).astype("float32")

    return pt, similarity_scores


def main():
    books, ratings = load_data()

    popular_df = build_popularity_model(books)
    pt, similarity_scores = build_collaborative_model(books, ratings)

    print("Saving models to /models ...")
    with open(f"{MODELS_DIR}/books.pkl", "wb") as f:
        pickle.dump(books, f)
    with open(f"{MODELS_DIR}/popular_df.pkl", "wb") as f:
        pickle.dump(popular_df, f)
    with open(f"{MODELS_DIR}/pt.pkl", "wb") as f:
        pickle.dump(pt, f)
    with open(f"{MODELS_DIR}/similarity_scores.pkl", "wb") as f:
        pickle.dump(similarity_scores, f)

    print("Done! Models saved:")
    print("  models/books.pkl")
    print("  models/popular_df.pkl")
    print("  models/pt.pkl")
    print("  models/similarity_scores.pkl")


if __name__ == "__main__":
    main()
