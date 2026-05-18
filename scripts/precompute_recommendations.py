import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
import pandas as pd
import faiss
import torch
import pickle
import yaml
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ranking_model import RankingMLP


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_genre_cols(movies):
    exclude = ["movieId", "title", "genres", "year", "title_clean"]
    return [c for c in movies.columns if c not in exclude]


def prepare_content_features(movies, genome):
    genre_cols = get_genre_cols(movies)
    genre_matrix = movies[["movieId"] + genre_cols].set_index("movieId").fillna(0)
    top_tag_cols = genome.var().nlargest(64).index
    genome_filtered = genome[top_tag_cols]
    genome_normalized = pd.DataFrame(
        normalize(genome_filtered.values, norm="l2"),
        index=genome_filtered.index,
        columns=genome_filtered.columns
    )
    return genre_matrix, genome_normalized


def retrieve_and_rank(user_idx, user_embeddings, faiss_index, item_embeddings,
                      ranking_model, movie_encoder, genre_matrix, genome_features, config):
    top_k_candidates = config["faiss"]["top_k"]
    top_k_final = config["ranking"]["top_k"]

    user_vector = user_embeddings[user_idx].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(user_vector)

    _, candidate_indices = faiss_index.search(user_vector, top_k_candidates)
    candidate_indices = candidate_indices[0]
    candidate_indices = candidate_indices[candidate_indices >= 0]

    candidate_movie_ids = movie_encoder.inverse_transform(candidate_indices)
    features = []
    for movie_idx, movie_id in zip(candidate_indices, candidate_movie_ids):
        user_emb = user_embeddings[user_idx]
        item_emb = item_embeddings[movie_idx]

        if movie_id in genre_matrix.index:
            genre_feat = genre_matrix.loc[movie_id].values.astype(np.float32)
        else:
            genre_feat = np.zeros(genre_matrix.shape[1], dtype=np.float32)

        if movie_id in genome_features.index:
            genome_feat = genome_features.loc[movie_id].values.astype(np.float32)
        else:
            genome_feat = np.zeros(genome_features.shape[1], dtype=np.float32)

        features.append(np.concatenate([user_emb, item_emb, genre_feat, genome_feat]))

    features = np.array(features, dtype=np.float32)

    with torch.no_grad():
        scores = ranking_model(torch.tensor(features).to("cpu")).numpy()

    ranked_indices = candidate_indices[np.argsort(scores)[::-1]]
    top_movie_indices = ranked_indices[:top_k_final]
    top_movie_ids = movie_encoder.inverse_transform(top_movie_indices)

    return top_movie_ids


def get_user_watch_history(user_idx, train, movies, n=10):
    user_history = train[
        (train["user_idx"] == user_idx) &
        (train["implicit_feedback"] == 1)
    ].sort_values("timestamp", ascending=False).head(n)

    history_movies = movies[movies["movieId"].isin(user_history["movieId"])][
        ["movieId", "title_clean", "genres", "year"]
    ]
    return history_movies


def sample_active_users(train, min_interactions=100, n_users=20):
    user_counts = train[train["implicit_feedback"] == 1].groupby("user_idx").size()
    active_users = user_counts[user_counts >= min_interactions].index.tolist()
    np.random.seed(42)
    sampled = np.random.choice(
        active_users, size=min(n_users, len(active_users)), replace=False
    )
    return sorted(sampled.tolist())


def precompute_recommendations(config, train, movies, genome, user_embeddings,
                                item_embeddings, faiss_index, ranking_model,
                                movie_encoder, out_dir):
    print("\nPrecomputing user recommendations...")

    genre_matrix, genome_features = prepare_content_features(movies, genome)
    sampled_users = sample_active_users(train, min_interactions=100, n_users=20)
    print(f"Selected {len(sampled_users)} users")

    output = {}
    for user_idx in tqdm(sampled_users, desc="Users"):
        try:
            recommended_ids = retrieve_and_rank(
                user_idx, user_embeddings, faiss_index,
                item_embeddings, ranking_model, movie_encoder,
                genre_matrix, genome_features, config
            )

            recommended_movies = movies[movies["movieId"].isin(recommended_ids)][
                ["movieId", "title_clean", "genres", "year"]
            ].copy()
            recommended_movies["year"] = recommended_movies["year"].fillna(0).astype(int)

            history = get_user_watch_history(user_idx, train, movies, n=10)
            history = history.copy()
            history["year"] = history["year"].fillna(0).astype(int)

            output[str(user_idx)] = {
                "user_idx": user_idx,
                "watch_history": history[["title_clean", "genres", "year"]].to_dict(orient="records"),
                "recommendations": recommended_movies[["title_clean", "genres", "year"]].to_dict(orient="records")
            }
        except Exception as e:
            print(f"Failed for user {user_idx}: {e}")
            continue

    out_path = os.path.join(out_dir, "sample_recommendations.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved recommendations for {len(output)} users")


def precompute_movies(movies, out_dir):
    print("\nPrecomputing movie data...")

    movies_out = movies[["movieId", "title_clean", "genres", "year"]].copy()
    movies_out["year"] = movies_out["year"].fillna(0).astype(int)
    movies_out = movies_out.dropna(subset=["title_clean"])
    movies_list = movies_out.to_dict(orient="records")

    with open(os.path.join(out_dir, "sample_movies.json"), "w") as f:
        json.dump(movies_list, f)
    print(f"Saved {len(movies_list)} movies")


def precompute_popular(movies, popularity, out_dir):
    print("\nPrecomputing popularity data...")

    exclude = ["movieId", "title", "genres", "year", "title_clean"]
    genre_cols = [c for c in movies.columns if c not in exclude
                  and c != "(no genres listed)"]

    popular_out = popularity[["movieId", "title_clean", "genres", "year",
                               "interaction_count", "final_score"]].copy()
    popular_out["year"] = popular_out["year"].fillna(0).astype(int)
    popular_out = popular_out.dropna(subset=["title_clean"])

    genre_flags = movies[["movieId"] + genre_cols].copy()
    popular_out = popular_out.merge(genre_flags, on="movieId", how="left")
    popular_list = popular_out.fillna(0).to_dict(orient="records")

    with open(os.path.join(out_dir, "sample_popular.json"), "w") as f:
        json.dump(popular_list, f)
    print(f"Saved {len(popular_list)} popular movies")


def precompute_ratings(train, out_dir):
    print("\nPrecomputing rating distribution...")

    rating_counts = train["rating"].value_counts().sort_index()
    ratings_out = {
        "ratings": rating_counts.index.tolist(),
        "counts": rating_counts.values.tolist()
    }

    with open(os.path.join(out_dir, "sample_ratings.json"), "w") as f:
        json.dump(ratings_out, f)
    print("Saved rating distribution")


def precompute_genre_data(movies, out_dir):
    print("\nPrecomputing genre data...")

    exclude = ["movieId", "title", "genres", "year", "title_clean"]
    genre_cols = [c for c in movies.columns if c not in exclude
                  and c != "(no genres listed)"]

    top_genres = movies[genre_cols].sum().sort_values(
        ascending=False
    ).head(10).index.tolist()

    genre_matrix_data = movies[top_genres].values
    cooccurrence = (genre_matrix_data.T @ genre_matrix_data).tolist()
    genre_counts = movies[genre_cols].sum().sort_values(ascending=False)

    genre_data_out = {
        "top_genres": top_genres,
        "cooccurrence": cooccurrence,
        "genre_counts": {g: int(c) for g, c in genre_counts.items()},
        "all_genres": genre_cols
    }

    with open(os.path.join(out_dir, "sample_genre_data.json"), "w") as f:
        json.dump(genre_data_out, f)
    print(f"Saved genre data for {len(top_genres)} top genres")


def precompute_similar_movies(movies, content_faiss_index, content_movie_ids,
                               content_matrix, out_dir, n_sample=100):
    print("\nPrecomputing similar movies...")

    sample_movie_ids = content_movie_ids[:n_sample]
    similar_out = {}

    for movie_id in tqdm(sample_movie_ids, desc="Movies"):
        idx = list(content_movie_ids).index(movie_id)
        query_vector = content_matrix[idx].reshape(1, -1).astype(np.float32)
        scores, indices = content_faiss_index.search(query_vector, 11)

        results = []
        for score, i in zip(scores[0], indices[0]):
            mid = int(content_movie_ids[i])
            if mid != int(movie_id):
                movie_row = movies[movies["movieId"] == mid]
                if len(movie_row) > 0:
                    results.append({
                        "movieId": mid,
                        "title_clean": movie_row.iloc[0]["title_clean"],
                        "genres": movie_row.iloc[0]["genres"],
                        "year": int(movie_row.iloc[0]["year"]) if not pd.isna(
                            movie_row.iloc[0]["year"]) else 0,
                        "similarity_score": float(score)
                    })

        if results:
            title_row = movies[movies["movieId"] == int(movie_id)]
            if len(title_row) > 0:
                title = title_row.iloc[0]["title_clean"]
                similar_out[title] = results[:10]

    with open(os.path.join(out_dir, "sample_similar_movies.json"), "w") as f:
        json.dump(similar_out, f)
    print(f"Saved similar movies for {len(similar_out)} sample movies")


def main():
    config = load_config()
    processed = config["paths"]["processed_data"]
    model_dir = config["paths"]["models"]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "app", "precomputed")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading artifacts...")
    train = pd.read_parquet(os.path.join(processed, "train.parquet"))
    movies = pd.read_parquet(os.path.join(processed, "movies.parquet"))
    genome = pd.read_parquet(os.path.join(processed, "genome_features.parquet"))
    popularity = pd.read_parquet(os.path.join(model_dir, "popularity_list.parquet"))

    user_embeddings = np.load(os.path.join(model_dir, "user_embeddings.npy"))
    item_embeddings = np.load(os.path.join(model_dir, "item_embeddings.npy"))
    faiss_index = faiss.read_index(os.path.join(model_dir, "faiss_index.bin"))
    content_faiss_index = faiss.read_index(
        os.path.join(model_dir, "content_faiss_index.bin")
    )
    content_movie_ids = np.load(os.path.join(model_dir, "content_movie_ids.npy"))
    content_matrix = np.load(os.path.join(model_dir, "content_matrix.npy"))

    with open(os.path.join(processed, "user_encoder.pkl"), "rb") as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(processed, "movie_encoder.pkl"), "rb") as f:
        movie_encoder = pickle.load(f)

    checkpoint = torch.load(
        os.path.join(model_dir, "ranking_mlp.pt"),
        map_location=torch.device("cpu")
    )
    ranking_model = RankingMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"]
    )
    ranking_model.load_state_dict(checkpoint["model_state_dict"])
    ranking_model.eval()

    precompute_recommendations(
        config, train, movies, genome, user_embeddings, item_embeddings,
        faiss_index, ranking_model, movie_encoder, out_dir
    )
    precompute_movies(movies, out_dir)
    precompute_popular(movies, popularity, out_dir)
    precompute_ratings(train, out_dir)
    precompute_genre_data(movies, out_dir)
    precompute_similar_movies(
        movies, content_faiss_index, content_movie_ids, content_matrix, out_dir
    )

    print("\nAll precomputed data saved.")
    print("\nFile sizes:")
    total_size = 0
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        size = os.path.getsize(fpath)
        total_size += size
        print(f"  {fname}: {size/1024:.1f} KB")
    print(f"  Total: {total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()