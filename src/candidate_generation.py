import os
import yaml
import numpy as np
import scipy.sparse as sp
import faiss
import implicit
import pickle
import mlflow
from tqdm import tqdm


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_interaction_matrix(config):
    processed = config["paths"]["processed_data"]
    print("Loading interaction matrix...")
    matrix = sp.load_npz(os.path.join(processed, "interaction_matrix.npz"))
    print(f"Matrix shape: {matrix.shape}")
    return matrix


def train_als(matrix, config):
    als_config = config["als"]
    print("\nTraining ALS model...")

    model = implicit.als.AlternatingLeastSquares(
        factors=als_config["factors"],
        iterations=als_config["iterations"],
        regularization=als_config["regularization"],
        alpha=als_config["alpha"],
        use_gpu=False
    )

    matrix_scaled = (matrix * als_config["alpha"]).astype(np.float32)
    model.fit(matrix_scaled)

    print("ALS training complete.")
    return model


def extract_embeddings(model):
    user_embeddings = np.array(model.user_factors)
    item_embeddings = np.array(model.item_factors)

    print(f"\nUser embeddings shape: {user_embeddings.shape}")
    print(f"Item embeddings shape: {item_embeddings.shape}")

    return user_embeddings, item_embeddings


def build_faiss_index(item_embeddings, config):
    faiss_config = config["faiss"]
    n_items, n_factors = item_embeddings.shape

    print(f"\nBuilding FAISS index over {n_items} items...")

    item_embeddings = item_embeddings.astype(np.float32)
    faiss.normalize_L2(item_embeddings)

    quantizer = faiss.IndexFlatIP(n_factors)
    index = faiss.IndexIVFFlat(quantizer, n_factors, faiss_config["n_lists"], faiss.METRIC_INNER_PRODUCT)

    index.train(item_embeddings)
    index.add(item_embeddings)
    index.nprobe = faiss_config["n_probe"]

    print(f"FAISS index built. Total vectors: {index.ntotal}")
    return index


def retrieve_candidates(user_idx, user_embeddings, index, config):
    top_k = config["faiss"]["top_k"]

    user_vector = user_embeddings[user_idx].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(user_vector)

    scores, indices = index.search(user_vector, top_k)
    return indices[0], scores[0]


def save_artifacts(config, model, user_embeddings, item_embeddings, index):
    model_dir = config["paths"]["models"]
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nSaving artifacts to {model_dir}...")

    with open(os.path.join(model_dir, "als_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    np.save(os.path.join(model_dir, "user_embeddings.npy"), user_embeddings)
    np.save(os.path.join(model_dir, "item_embeddings.npy"), item_embeddings)

    faiss.write_index(index, os.path.join(model_dir, "faiss_index.bin"))

    print("All artifacts saved.")


def run_candidate_generation():
    config = load_config()

    mlflow.set_tracking_uri(config["paths"]["mlruns"])
    mlflow.set_experiment("candidate_generation")

    with mlflow.start_run(run_name="als_faiss"):
        mlflow.log_params({
            "factors": config["als"]["factors"],
            "iterations": config["als"]["iterations"],
            "regularization": config["als"]["regularization"],
            "alpha": config["als"]["alpha"],
            "faiss_n_lists": config["faiss"]["n_lists"],
            "faiss_n_probe": config["faiss"]["n_probe"],
            "top_k": config["faiss"]["top_k"]
        })

        matrix = load_interaction_matrix(config)
        model = train_als(matrix, config)
        user_embeddings, item_embeddings = extract_embeddings(model)
        index = build_faiss_index(item_embeddings, config)

        mlflow.log_metric("n_users", user_embeddings.shape[0])
        mlflow.log_metric("n_items", item_embeddings.shape[0])
        mlflow.log_metric("faiss_index_size", index.ntotal)

        save_artifacts(config, model, user_embeddings, item_embeddings, index)

        print("\nCandidate generation complete.")


if __name__ == "__main__":
    run_candidate_generation()