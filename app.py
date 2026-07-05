"""
Book Recommendation System - Flask app
----------------------------------------
Two recommendation strategies, both pre-computed by model_builder.py:

  1. Popularity-based  -> shown on the home page ("/")
  2. Item-based collaborative filtering -> "/recommend"

Run:
    python model_builder.py   # once, to (re)generate the pickle files
    python app.py
"""

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------- models --
books = pickle.load(open("models/books.pkl", "rb"))
popular_df = pickle.load(open("models/popular_df.pkl", "rb"))
pt = pickle.load(open("models/pt.pkl", "rb"))
similarity_scores = pickle.load(open("models/similarity_scores.pkl", "rb"))

# A sorted list of every book title the CF model knows about, used to
# power the autocomplete on the recommend page.
KNOWN_TITLES = sorted(pt.index.tolist())


def get_book_meta(title):
    """Look up author / rating / cover for a title from the books table."""
    row = books[books["title"] == title]
    if row.empty:
        return {
            "title": title,
            "authors": "Unknown",
            "average_rating": None,
            "image_url": "https://images.gr-assets.com/books/1310220028s/1.jpg",
        }
    row = row.iloc[0]
    return {
        "title": row["title"],
        "authors": row["authors"],
        "average_rating": round(float(row["average_rating"]), 2),
        "image_url": row["image_url"],
    }


@app.route("/")
def index():
    top_books = [get_book_meta(t) for t in popular_df["title"].tolist()]
    return render_template("index.html", books=top_books)


@app.route("/recommend")
def recommend_form():
    return render_template("recommend.html", titles=KNOWN_TITLES, searched=None, results=None)


@app.route("/recommend_books", methods=["POST"])
def recommend_books():
    user_input = request.form.get("book_title", "").strip()

    if user_input not in pt.index:
        return render_template(
            "recommend.html",
            titles=KNOWN_TITLES,
            searched=user_input,
            results=[],
            not_found=True,
        )

    idx = np.where(pt.index == user_input)[0][0]
    scores = list(enumerate(similarity_scores[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  # top 5, skip itself

    results = [get_book_meta(pt.index[i]) for i, _ in scores]

    return render_template(
        "recommend.html",
        titles=KNOWN_TITLES,
        searched=user_input,
        results=results,
        not_found=False,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
