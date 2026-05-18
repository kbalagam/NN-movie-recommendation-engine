import os
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config):
    raw = config["paths"]["raw_data"]
    print("Loading raw data...")

    ratings = pd.read_csv(os.path.join(raw, "rating.csv"))
    movies = pd.read_csv(os.path.join(raw, "movie.csv"))
    tags = pd.read_csv(os.path.join(raw, "tag.csv"))
    genome_scores = pd.read_csv(os.path.join(raw, "genome_scores.csv"))
    genome_tags = pd.read_csv(os.path.join(raw, "genome_tags.csv"))
    links = pd.read_csv(os.path.join(raw, "link.csv"))

    print(f"Ratings     : {ratings.shape}")
    print(f"Movies      : {movies.shape}")
    print(f"Tags        : {tags.shape}")
    print(f"Genome scores: {genome_scores.shape}")
    print(f"Genome tags : {genome_tags.shape}")
    print(f"Links       : {links.shape}")

    return ratings, movies, tags, genome_scores, genome_tags, links


def filter_interactions(ratings, config):
    min_user = config["data"]["min_user_interactions"]
    min_item = config["data"]["min_item_interactions"]

    print(f"\nFiltering users with < {min_user} interactions and items with < {min_item} interactions...")

    while True:
        user_counts = ratings["userId"].value_counts()
        item_counts = ratings["movieId"].value_counts()

        valid_users = user_counts[user_counts >= min_user].index
        valid_items = item_counts[item_counts >= min_item].index

        filtered = ratings[ratings["userId"].isin(valid_users) & ratings["movieId"].isin(valid_items)]

        if len(filtered) == len(ratings):
            break
        ratings = filtered

    print(f"Filtered ratings shape: {ratings.shape}")
    print(f"Unique users: {ratings['userId'].nunique()}")
    print(f"Unique movies: {ratings['movieId'].nunique()}")
    return ratings


def convert_to_implicit(ratings, config):
    threshold = config["data"]["implicit_threshold"]
    print(f"\nConverting to implicit feedback (threshold={threshold})...")

    ratings = ratings.copy()
    ratings["implicit_feedback"] = (ratings["rating"] >= threshold).astype(float)

    above = ratings["implicit_feedback"].sum()
    print(f"Positive interactions: {int(above)} / {len(ratings)} ({100*above/len(ratings):.1f}%)")

    return ratings


def encode_ids(ratings):
    print("\nEncoding user and movie IDs...")

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings = ratings.copy()
    ratings["user_idx"] = user_encoder.fit_transform(ratings["userId"])
    ratings["movie_idx"] = movie_encoder.fit_transform(ratings["movieId"])

    n_users = ratings["user_idx"].nunique()
    n_movies = ratings["movie_idx"].nunique()

    print(f"Encoded users : {n_users}")
    print(f"Encoded movies: {n_movies}")

    return ratings, user_encoder, movie_encoder


def build_interaction_matrix(ratings):
    print("\nBuilding sparse user-item interaction matrix...")

    n_users = ratings["user_idx"].nunique()
    n_movies = ratings["movie_idx"].nunique()

    matrix = sp.csr_matrix(
        (ratings["implicit_feedback"].values,
         (ratings["user_idx"].values, ratings["movie_idx"].values)),
        shape=(n_users, n_movies)
    )

    sparsity = 1 - matrix.nnz / (n_users * n_movies)
    print(f"Matrix shape : {matrix.shape}")
    print(f"Non-zero entries: {matrix.nnz}")
    print(f"Sparsity     : {sparsity:.4%}")

    return matrix


def process_movies(movies):
    print("\nProcessing movie metadata...")

    movies = movies.copy()
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

    genres = movies["genres"].str.get_dummies(sep="|")
    movies = pd.concat([movies, genres], axis=1)

    print(f"Genres found: {list(genres.columns)}")
    return movies


def process_genome(genome_scores, genome_tags):
    print("\nProcessing genome tag scores...")

    genome = genome_scores.merge(genome_tags, on="tagId")
    top_tags = genome.groupby("tagId")["relevance"].mean().nlargest(128).index
    genome_filtered = genome[genome["tagId"].isin(top_tags)]

    genome_pivot = genome_filtered.pivot_table(
        index="movieId", columns="tagId", values="relevance", fill_value=0
    )

    print(f"Genome feature matrix shape: {genome_pivot.shape}")
    return genome_pivot


def train_test_split_temporal(ratings, config):
    print("\nSplitting train/test temporally...")

    ratings = ratings.sort_values("timestamp")
    split_idx = int(len(ratings) * (1 - config["data"]["test_size"]))

    train = ratings.iloc[:split_idx]
    test = ratings.iloc[split_idx:]

    print(f"Train size: {len(train)}")
    print(f"Test size : {len(test)}")

    return train, test


def save_processed(config, train, test, matrix, user_encoder, movie_encoder, movies, genome_pivot):
    out = config["paths"]["processed_data"]
    os.makedirs(out, exist_ok=True)
    print(f"\nSaving processed files to {out}...")

    train.to_parquet(os.path.join(out, "train.parquet"), index=False)
    test.to_parquet(os.path.join(out, "test.parquet"), index=False)
    movies.to_parquet(os.path.join(out, "movies.parquet"), index=False)
    genome_pivot.to_parquet(os.path.join(out, "genome_features.parquet"))

    sp.save_npz(os.path.join(out, "interaction_matrix.npz"), matrix)

    with open(os.path.join(out, "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(out, "movie_encoder.pkl"), "wb") as f:
        pickle.dump(movie_encoder, f)

    print("All files saved successfully.")


def run_pipeline():
    config = load_config()

    ratings, movies, tags, genome_scores, genome_tags, links = load_raw_data(config)
    ratings = filter_interactions(ratings, config)
    ratings = convert_to_implicit(ratings, config)
    ratings, user_encoder, movie_encoder = encode_ids(ratings)
    matrix = build_interaction_matrix(ratings)
    movies = process_movies(movies)
    genome_pivot = process_genome(genome_scores, genome_tags)
    train, test = train_test_split_temporal(ratings, config)

    save_processed(config, train, test, matrix, user_encoder, movie_encoder, movies, genome_pivot)

    print("\nData pipeline complete.")
    return config


if __name__ == "__main__":
    run_pipeline()