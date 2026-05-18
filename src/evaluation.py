import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import faiss
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_artifacts(config):
    processed = config["paths"]["processed_data"]
    model_dir = config["paths"]["models"]

    print("Loading artifacts...")

    test = pd.read_parquet(os.path.join(processed, "test.parquet"))
    movies = pd.read_parquet(os.path.join(processed, "movies.parquet"))

    user_embeddings = np.load(os.path.join(model_dir, "user_embeddings.npy"))
    item_embeddings = np.load(os.path.join(model_dir, "item_embeddings.npy"))

    faiss_index = faiss.read_index(os.path.join(model_dir, "faiss_index.bin"))

    with open(os.path.join(processed, "user_encoder.pkl"), "rb") as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(processed, "movie_encoder.pkl"), "rb") as f:
        movie_encoder = pickle.load(f)

    checkpoint = torch.load(
        os.path.join(model_dir, "ranking_mlp.pt"),
        map_location=torch.device("cpu")
    )

    return (test, movies, user_embeddings, item_embeddings,
            faiss_index, user_encoder, movie_encoder, checkpoint)


def load_ranking_model(checkpoint):
    from ranking_model import RankingMLP

    model = RankingMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to("cpu")
    return model


def get_genre_cols(movies):
    exclude = ["movieId", "title", "genres", "year", "title_clean"]
    return [c for c in movies.columns if c not in exclude]


def prepare_content_features(movies, config):
    processed = config["paths"]["processed_data"]
    genome = pd.read_parquet(os.path.join(processed, "genome_features.parquet"))

    genre_cols = get_genre_cols(movies)
    genre_matrix = movies[["movieId"] + genre_cols].set_index("movieId").fillna(0)

    top_tag_cols = genome.var().nlargest(64).index
    genome = genome[top_tag_cols]

    from sklearn.preprocessing import normalize
    genome_normalized = pd.DataFrame(
        normalize(genome.values, norm="l2"),
        index=genome.index,
        columns=genome.columns
    )

    return genre_matrix, genome_normalized


def build_ranking_features(user_idx, candidate_indices, user_embeddings,
                            item_embeddings, movie_encoder, genre_matrix,
                            genome_features):
    user_emb = user_embeddings[user_idx]
    candidate_movie_ids = movie_encoder.inverse_transform(candidate_indices)

    features = []
    for movie_idx, movie_id in zip(candidate_indices, candidate_movie_ids):
        item_emb = item_embeddings[movie_idx]

        if movie_id in genre_matrix.index:
            genre_feat = genre_matrix.loc[movie_id].values.astype(np.float32)
        else:
            genre_feat = np.zeros(genre_matrix.shape[1], dtype=np.float32)

        if movie_id in genome_features.index:
            genome_feat = genome_features.loc[movie_id].values.astype(np.float32)
        else:
            genome_feat = np.zeros(genome_features.shape[1], dtype=np.float32)

        feature_vector = np.concatenate([user_emb, item_emb, genre_feat, genome_feat])
        features.append(feature_vector)

    return np.array(features, dtype=np.float32)


def retrieve_and_rank(user_idx, user_embeddings, faiss_index, item_embeddings,
                      ranking_model, movie_encoder, genre_matrix, genome_features, config):
    top_k_candidates = config["faiss"]["top_k"]
    top_k_final = config["ranking"]["top_k"]

    user_vector = user_embeddings[user_idx].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(user_vector)

    _, candidate_indices = faiss_index.search(user_vector, top_k_candidates)
    candidate_indices = candidate_indices[0]
    candidate_indices = candidate_indices[candidate_indices >= 0]

    features = build_ranking_features(
        user_idx, candidate_indices, user_embeddings,
        item_embeddings, movie_encoder, genre_matrix, genome_features
    )

    with torch.no_grad():
        scores = ranking_model(torch.tensor(features).to("cpu")).numpy()

    ranked_indices = candidate_indices[np.argsort(scores)[::-1]]
    top_movie_indices = ranked_indices[:top_k_final]
    top_movie_ids = movie_encoder.inverse_transform(top_movie_indices)

    return top_movie_ids


def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate(config, n_users=1000):
    (test, movies, user_embeddings, item_embeddings,
     faiss_index, user_encoder, movie_encoder, checkpoint) = load_artifacts(config)

    ranking_model = load_ranking_model(checkpoint)
    genre_matrix, genome_features = prepare_content_features(movies, config)

    test_positive = test[test["implicit_feedback"] == 1]
    user_item_test = (
        test_positive.groupby("user_idx")["movieId"]
        .apply(list)
        .to_dict()
    )

    valid_users = [u for u in user_item_test.keys() if u < len(user_embeddings)]
    np.random.seed(config["data"]["random_seed"])
    sampled_users = np.random.choice(
        valid_users, size=min(n_users, len(valid_users)), replace=False
    )

    k_values = config["evaluation"]["k_values"]
    metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

    print(f"\nEvaluating on {len(sampled_users)} users...")

    for user_idx in tqdm(sampled_users, desc="Evaluating"):
        relevant_movies = user_item_test.get(user_idx, [])
        if len(relevant_movies) == 0:
            continue

        try:
            recommended = retrieve_and_rank(
                user_idx, user_embeddings, faiss_index, item_embeddings,
                ranking_model, movie_encoder, genre_matrix, genome_features, config
            )
        except Exception:
            continue

        for k in k_values:
            metrics[k]["precision"].append(precision_at_k(recommended, relevant_movies, k))
            metrics[k]["recall"].append(recall_at_k(recommended, relevant_movies, k))
            metrics[k]["ndcg"].append(ndcg_at_k(recommended, relevant_movies, k))

    print("\nEvaluation Results:")
    print("-" * 45)

    results = {}
    for k in k_values:
        p = np.mean(metrics[k]["precision"])
        r = np.mean(metrics[k]["recall"])
        n = np.mean(metrics[k]["ndcg"])

        results[k] = {"precision": p, "recall": r, "ndcg": n}

        print(f"K={k:2d} | Precision: {p:.4f} | Recall: {r:.4f} | NDCG: {n:.4f}")

        mlflow.log_metric(f"precision_at_{k}", p)
        mlflow.log_metric(f"recall_at_{k}", r)
        mlflow.log_metric(f"ndcg_at_{k}", n)

    print("-" * 45)
    return results


def run_evaluation():
    config = load_config()

    mlflow.set_tracking_uri(config["paths"]["mlruns"])
    mlflow.set_experiment("evaluation")

    with mlflow.start_run(run_name="two_stage_eval"):
        results = evaluate(config, n_users=1000)

    print("\nEvaluation complete.")
    return results


if __name__ == "__main__":
    run_evaluation()