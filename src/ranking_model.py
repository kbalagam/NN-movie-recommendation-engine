import os
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from tqdm import tqdm


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_artifacts(config):
    processed = config["paths"]["processed_data"]
    model_dir = config["paths"]["models"]

    print("Loading artifacts...")

    train = pd.read_parquet(os.path.join(processed, "train.parquet"))
    movies = pd.read_parquet(os.path.join(processed, "movies.parquet"))
    genome = pd.read_parquet(os.path.join(processed, "genome_features.parquet"))

    user_embeddings = np.load(os.path.join(model_dir, "user_embeddings.npy"))
    item_embeddings = np.load(os.path.join(model_dir, "item_embeddings.npy"))

    with open(os.path.join(processed, "user_encoder.pkl"), "rb") as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(processed, "movie_encoder.pkl"), "rb") as f:
        movie_encoder = pickle.load(f)

    return train, movies, genome, user_embeddings, item_embeddings, user_encoder, movie_encoder


def prepare_genre_features(movies):
    genre_cols = [c for c in movies.columns if c not in
                  ["movieId", "title", "genres", "year", "title_clean"]]
    genre_matrix = movies[["movieId"] + genre_cols].set_index("movieId")
    genre_matrix = genre_matrix.fillna(0)
    return genre_matrix, genre_cols


def prepare_genome_features(genome, movie_encoder):
    known_movie_ids = set(movie_encoder.classes_)
    genome = genome[genome.index.isin(known_movie_ids)]
    
    top_tag_cols = genome.var().nlargest(64).index
    genome = genome[top_tag_cols]
    
    genome_normalized = pd.DataFrame(
        normalize(genome.values, norm="l2"),
        index=genome.index,
        columns=genome.columns
    )
    return genome_normalized


def build_samples(train, user_embeddings, item_embeddings, movie_encoder,
                  genre_matrix, genome_features, n_negatives=4):
    print("\nBuilding training samples...")

    config = load_config()
    max_samples = config["data"].get("max_training_samples", 2000000)

    positive_rows = train[train["implicit_feedback"] == 1].copy()

    if len(positive_rows) > max_samples:
        positive_rows = positive_rows.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} positives from {train['implicit_feedback'].sum():.0f} total")

    n_positives = len(positive_rows)
    n_items = len(movie_encoder.classes_)

    print(f"Positive interactions: {n_positives}")

    genre_cols = genre_matrix.columns.tolist()
    n_genre = len(genre_cols)
    n_genome = genome_features.shape[1]

    all_movie_ids = movie_encoder.classes_

    genre_lookup = np.zeros((n_items, n_genre), dtype=np.float32)
    genome_lookup = np.zeros((n_items, n_genome), dtype=np.float32)

    print("Building feature lookup tables...")
    for i, movie_id in enumerate(tqdm(all_movie_ids, desc="Feature lookup")):
        if movie_id in genre_matrix.index:
            genre_lookup[i] = genre_matrix.loc[movie_id].values.astype(np.float32)
        if movie_id in genome_features.index:
            genome_lookup[i] = genome_features.loc[movie_id].values.astype(np.float32)

    user_indices = positive_rows["user_idx"].values.astype(np.int32)
    item_indices = positive_rows["movie_idx"].values.astype(np.int32)

    print("Extracting positive sample features...")
    pos_user_embs = user_embeddings[user_indices]
    pos_item_embs = item_embeddings[item_indices]
    pos_genre = genre_lookup[item_indices]
    pos_genome = genome_lookup[item_indices]
    pos_labels = np.ones(n_positives, dtype=np.float32)

    print("Sampling negatives...")
    neg_item_indices = np.random.randint(0, n_items, size=(n_positives, n_negatives))

    neg_user_embs = np.repeat(pos_user_embs, n_negatives, axis=0)
    neg_item_embs = item_embeddings[neg_item_indices.ravel()]
    neg_genre = genre_lookup[neg_item_indices.ravel()]
    neg_genome = genome_lookup[neg_item_indices.ravel()]
    neg_labels = np.zeros(n_positives * n_negatives, dtype=np.float32)

    print("Concatenating all samples...")
    all_user_embs = np.concatenate([pos_user_embs, neg_user_embs], axis=0)
    all_item_embs = np.concatenate([pos_item_embs, neg_item_embs], axis=0)
    all_genre = np.concatenate([pos_genre, neg_genre], axis=0)
    all_genome = np.concatenate([pos_genome, neg_genome], axis=0)
    all_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    shuffle_idx = np.random.permutation(len(all_labels))
    all_user_embs = all_user_embs[shuffle_idx]
    all_item_embs = all_item_embs[shuffle_idx]
    all_genre = all_genre[shuffle_idx]
    all_genome = all_genome[shuffle_idx]
    all_labels = all_labels[shuffle_idx]

    print(f"Total samples: {len(all_labels)}")

    return all_user_embs, all_item_embs, all_genre, all_genome, all_labels


class RankingDataset(Dataset):
    def __init__(self, user_embs, item_embs, genre_feats, genome_feats, labels):
        self.features = np.concatenate(
            [user_embs, item_embs, genre_feats, genome_feats], axis=1
        ).astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class RankingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(RankingMLP, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_model(samples, config, device):
    ranking_config = config["ranking"]

    user_embs, item_embs, genre_feats, genome_feats, labels = samples
    dataset = RankingDataset(user_embs, item_embs, genre_feats, genome_feats, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=ranking_config["batch_size"],
        shuffle=True,
        num_workers=0
    )

    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]
    print(f"\nModel input dimension: {input_dim}")

    model = RankingMLP(
        input_dim=input_dim,
        hidden_dims=ranking_config["hidden_dims"],
        dropout=ranking_config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ranking_config["learning_rate"])
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print("\nTraining ranking model...")
    for epoch in range(ranking_config["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{ranking_config['epochs']}"):
            features = features.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (predictions >= 0.5).float()
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", accuracy, step=epoch)

    return model, input_dim


def save_model(model, input_dim, config):
    model_dir = config["paths"]["models"]
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "ranking_mlp.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": config["ranking"]["hidden_dims"],
        "dropout": config["ranking"]["dropout"]
    }, model_path)

    print(f"\nRanking model saved to {model_path}")


def run_ranking_model():
    config = load_config()
    device = get_device()

    mlflow.set_tracking_uri(config["paths"]["mlruns"])
    mlflow.set_experiment("ranking_model")

    with mlflow.start_run(run_name="mlp_ranker"):
        mlflow.log_params({
            "hidden_dims": str(config["ranking"]["hidden_dims"]),
            "dropout": config["ranking"]["dropout"],
            "learning_rate": config["ranking"]["learning_rate"],
            "batch_size": config["ranking"]["batch_size"],
            "epochs": config["ranking"]["epochs"]
        })

        train, movies, genome, user_embeddings, item_embeddings, user_encoder, movie_encoder = load_artifacts(config)

        genre_matrix, genre_cols = prepare_genre_features(movies)
        genome_features = prepare_genome_features(genome, movie_encoder)

        mlflow.log_param("n_genre_features", len(genre_cols))
        mlflow.log_param("n_genome_features", genome_features.shape[1])

        user_embs, item_embs, genre_feats, genome_feats, labels = build_samples(
            train, user_embeddings, item_embeddings,
            movie_encoder, genre_matrix, genome_features
        )

        model, input_dim = train_model(
            (user_embs, item_embs, genre_feats, genome_feats, labels), config, device
        )

        save_model(model, input_dim, config)

        print("\nRanking model training complete.")


if __name__ == "__main__":
    run_ranking_model()