import os
import yaml
import numpy as np
import pandas as pd
import pickle
import faiss
from sklearn.preprocessing import normalize


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_artifacts(config):
    processed = config["paths"]["processed_data"]

    train = pd.read_parquet(os.path.join(processed, "train.parquet"))
    movies = pd.read_parquet(os.path.join(processed, "movies.parquet"))
    genome = pd.read_parquet(os.path.join(processed, "genome_features.parquet"))

    with open(os.path.join(processed, "movie_encoder.pkl"), "rb") as f:
        movie_encoder = pickle.load(f)

    return train, movies, genome, movie_encoder


def get_genre_cols(movies):
    exclude = ["movieId", "title", "genres", "year", "title_clean"]
    return [c for c in movies.columns if c not in exclude]


def build_popularity_list(train, movies, config):
    print("Building recency-weighted popularity list...")

    top_k = config["cold_start"]["popularity_top_k"]

    interactions = train[train["implicit_feedback"] == 1].copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")
    interactions = interactions.dropna(subset=["timestamp"])

    print(f"Interactions after timestamp fix: {len(interactions)}")

    max_ts = interactions["timestamp"].max()
    min_ts = interactions["timestamp"].min()

    time_range = (max_ts - min_ts).total_seconds()
    interactions["recency_weight"] = (
        (interactions["timestamp"] - min_ts).dt.total_seconds() / time_range
    ).clip(0.1, 1.0)

    popularity = (
        interactions.groupby("movieId")
        .agg(
            interaction_count=("userId", "nunique"),
            weighted_score=("recency_weight", "sum")
        )
        .reset_index()
    )

    max_count = float(popularity["interaction_count"].max())
    max_score = float(popularity["weighted_score"].max())

    popularity["final_score"] = (
        0.5 * popularity["interaction_count"].astype(float) / max_count +
        0.5 * popularity["weighted_score"].astype(float) / max_score
    )

    popularity = popularity.sort_values("final_score", ascending=False)

    movies_subset = movies[["movieId", "title_clean", "genres", "year"]].drop_duplicates("movieId")

    popularity = popularity.merge(movies_subset, on="movieId", how="left")

    print(f"Popularity shape after merge: {popularity.shape}")
    print(f"Null title_clean: {popularity['title_clean'].isna().sum()}")

    popularity = popularity.head(top_k)

    model_dir = config["paths"]["models"]
    os.makedirs(model_dir, exist_ok=True)
    popularity.to_parquet(os.path.join(model_dir, "popularity_list.parquet"), index=False)

    print(f"Popularity list built. Top {top_k} movies saved.")
    return popularity


def build_content_index(movies, genome, movie_encoder, config):
    print("Building FAISS content index...")

    genre_cols = get_genre_cols(movies)
    genre_matrix = movies[["movieId"] + genre_cols].set_index("movieId").fillna(0)

    known_movie_ids = set(movie_encoder.classes_)
    genome = genome[genome.index.isin(known_movie_ids)]

    top_tag_cols = genome.var().nlargest(64).index
    genome = genome[top_tag_cols]
    genome_normalized = pd.DataFrame(
        normalize(genome.values, norm="l2"),
        index=genome.index,
        columns=genome.columns
    )

    common_movies = list(
        set(genre_matrix.index) & set(genome_normalized.index) & known_movie_ids
    )
    print(f"Movies with full content features: {len(common_movies)}")

    genre_sub = genre_matrix.loc[common_movies].values.astype(np.float32)
    genome_sub = genome_normalized.loc[common_movies].values.astype(np.float32)

    content_matrix = np.concatenate([genre_sub, genome_sub], axis=1)
    content_matrix = normalize(content_matrix, norm="l2").astype(np.float32)

    movie_ids_array = np.array(common_movies)

    dimension = content_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(content_matrix)

    model_dir = config["paths"]["models"]

    faiss.write_index(index, os.path.join(model_dir, "content_faiss_index.bin"))
    np.save(os.path.join(model_dir, "content_movie_ids.npy"), movie_ids_array)
    np.save(os.path.join(model_dir, "content_matrix.npy"), content_matrix)

    movie_meta = movies[movies["movieId"].isin(common_movies)][
        ["movieId", "title_clean", "genres", "year"]
    ].set_index("movieId")
    movie_meta.to_parquet(os.path.join(model_dir, "movie_meta.parquet"))

    print(f"FAISS content index built. Vectors: {index.ntotal}")
    return index, movie_ids_array, content_matrix, movie_meta


def get_popular_recommendations(config, genre_filter=None, top_k=10):
    model_dir = config["paths"]["models"]
    popularity = pd.read_parquet(os.path.join(model_dir, "popularity_list.parquet"))

    if genre_filter:
        mask = popularity["genres"].apply(
            lambda g: any(genre.lower() in str(g).lower() for genre in genre_filter)
        )
        filtered = popularity[mask]
        if len(filtered) >= top_k:
            popularity = filtered

    return popularity.head(top_k)


def get_similar_movies(movie_id, config, top_k=10):
    model_dir = config["paths"]["models"]

    index = faiss.read_index(os.path.join(model_dir, "content_faiss_index.bin"))
    movie_ids_array = np.load(os.path.join(model_dir, "content_movie_ids.npy"))
    content_matrix = np.load(os.path.join(model_dir, "content_matrix.npy"))
    movie_meta = pd.read_parquet(os.path.join(model_dir, "movie_meta.parquet"))

    movie_id_list = list(movie_ids_array)
    if movie_id not in movie_id_list:
        print(f"Movie {movie_id} not found in content index.")
        return pd.DataFrame()

    idx = movie_id_list.index(movie_id)
    query_vector = content_matrix[idx].reshape(1, -1).astype(np.float32)

    scores, indices = index.search(query_vector, top_k + 1)

    result_ids = []
    result_scores = []
    for score, i in zip(scores[0], indices[0]):
        mid = movie_ids_array[i]
        if mid != movie_id:
            result_ids.append(mid)
            result_scores.append(score)

    result_ids = result_ids[:top_k]
    result_scores = result_scores[:top_k]

    result = movie_meta.loc[result_ids].copy()
    result["similarity_score"] = result_scores

    return result


def recommend_for_new_user(config, genre_filter=None, top_k=10):
    return get_popular_recommendations(config, genre_filter=genre_filter, top_k=top_k)


def run_cold_start():
    config = load_config()
    train, movies, genome, movie_encoder = load_artifacts(config)

    popularity = build_popularity_list(train, movies, config)
    index, movie_ids_array, content_matrix, movie_meta = build_content_index(
        movies, genome, movie_encoder, config
    )

    print("\nTop 5 popular movies:")
    print(popularity[["title_clean", "interaction_count", "final_score"]].head())

    sample_movie_id = movie_ids_array[0]
    sample_title = movie_meta.loc[sample_movie_id, "title_clean"]
    print(f"\nMovies similar to '{sample_title}':")
    similar = get_similar_movies(sample_movie_id, config, top_k=5)
    print(similar[["title_clean", "similarity_score"]])

    print("\nCold start module complete.")


if __name__ == "__main__":
    run_cold_start()