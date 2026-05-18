import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

import sys
import json
import numpy as np
import pandas as pd
import faiss
import torch
import pickle
import yaml
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logger import log_interaction, log_recommendations, get_recent_interactions


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_precomputed_path(filename):
    return os.path.join(os.path.dirname(__file__), "precomputed", filename)


def load_precomputed(filename):
    path = get_precomputed_path(filename)
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource
def load_all_artifacts():
    config = load_config()
    processed = config["paths"]["processed_data"]
    model_dir = config["paths"]["models"]

    live_mode = os.path.exists(os.path.join(model_dir, "faiss_index.bin"))

    if live_mode:
        from src.ranking_model import RankingMLP

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
        ranking_model = RankingMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"]
        )
        ranking_model.load_state_dict(checkpoint["model_state_dict"])
        ranking_model.eval()

        train = pd.read_parquet(os.path.join(processed, "train.parquet"))
        movies = pd.read_parquet(os.path.join(processed, "movies.parquet"))
        genome = pd.read_parquet(os.path.join(processed, "genome_features.parquet"))
        popularity = pd.read_parquet(os.path.join(model_dir, "popularity_list.parquet"))
        movie_meta = pd.read_parquet(os.path.join(model_dir, "movie_meta.parquet"))
        content_faiss_index = faiss.read_index(
            os.path.join(model_dir, "content_faiss_index.bin")
        )
        content_movie_ids = np.load(os.path.join(model_dir, "content_movie_ids.npy"))
        content_matrix = np.load(os.path.join(model_dir, "content_matrix.npy"))

        return (config, live_mode, user_embeddings, item_embeddings, faiss_index,
                user_encoder, movie_encoder, ranking_model, train, movies,
                genome, popularity, movie_meta, content_faiss_index,
                content_movie_ids, content_matrix)
    else:
        return (config, live_mode, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None)


def get_genre_cols(movies):
    exclude = ["movieId", "title", "genres", "year", "title_clean"]
    return [c for c in movies.columns if c not in exclude]


def prepare_content_features(movies, genome):
    from sklearn.preprocessing import normalize
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
                      ranking_model, movie_encoder, genre_matrix, genome_features,
                      config, movies, freshness_bonus=0.0):
    from src.ranking_model import RankingMLP
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
        mlp_scores = ranking_model(torch.tensor(features).to("cpu")).numpy()

    if freshness_bonus > 0.0:
        movie_year_map = movies[["movieId", "year"]].dropna().set_index("movieId")["year"].to_dict()
        years = np.array([movie_year_map.get(mid, np.nan) for mid in candidate_movie_ids])
        valid_years = years[~np.isnan(years)]
        if len(valid_years) > 0:
            min_year = valid_years.min()
            max_year = valid_years.max()
            year_range = max_year - min_year if max_year != min_year else 1.0
            recency_weights = np.where(
                np.isnan(years), 0.0, (years - min_year) / year_range
            )
            mlp_scores = mlp_scores + freshness_bonus * recency_weights

    ranked_indices = candidate_indices[np.argsort(mlp_scores)[::-1]]
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


def sample_active_users(train, min_interactions=50, n_users=500):
    user_counts = train[train["implicit_feedback"] == 1].groupby("user_idx").size()
    active_users = user_counts[user_counts >= min_interactions].index.tolist()
    np.random.seed(42)
    sampled = np.random.choice(
        active_users, size=min(n_users, len(active_users)), replace=False
    )
    return sorted(sampled.tolist())


def render_movie_table(df, score_col=None, score_label="Score"):
    display_cols = ["title_clean", "genres", "year"]
    rename = {"title_clean": "Title", "genres": "Genres", "year": "Year"}

    if score_col and score_col in df.columns:
        display_cols.append(score_col)
        rename[score_col] = score_label

    display_df = df[display_cols].rename(columns=rename).reset_index(drop=True)
    display_df.index += 1

    if "Year" in display_df.columns:
        display_df["Year"] = display_df["Year"].fillna(0).astype(int).replace(0, "Unknown")

    st.dataframe(display_df, use_container_width=True)


def generate_architecture_image(arch_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    def draw_box(ax, x, y, w, h, label, sublabel=None, facecolor="#F5F5F5",
                 edgecolor="#AAAAAA", labelcolor="#222222", sublabelcolor="#555555",
                 fontsize=10, subfontsize=9):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=facecolor,
                             edgecolor=edgecolor,
                             linewidth=1.2)
        ax.add_patch(box)
        if sublabel:
            ax.text(x + w / 2, y + h * 0.62, label,
                    ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=labelcolor)
            ax.text(x + w / 2, y + h * 0.28, sublabel,
                    ha="center", va="center",
                    fontsize=subfontsize, color=sublabelcolor)
        else:
            ax.text(x + w / 2, y + h / 2, label,
                    ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=labelcolor)

    def draw_container(ax, x, y, w, h, label, edgecolor, facecolor):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=facecolor,
                             edgecolor=edgecolor,
                             linewidth=1.2,
                             linestyle="dashed")
        ax.add_patch(box)
        ax.text(x + 0.15, y + h - 0.18, label,
                ha="left", va="center",
                fontsize=8.5, fontweight="bold", color=edgecolor)

    def draw_arrow(ax, x1, y1, x2, y2, color="#888888"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))

    draw_container(ax, 0.2, 2.2, 15.6, 3.5, "MAIN PIPELINE",
                   edgecolor="#AAAAAA", facecolor="#F9F9F9")
    draw_box(ax, 0.4, 2.6, 1.8, 1.4, "MovieLens 20M",
             sublabel="138K users · 27K movies",
             facecolor="#F5F5F5", edgecolor="#AAAAAA",
             labelcolor="#222222", sublabelcolor="#555555")
    draw_arrow(ax, 2.2, 3.3, 2.6, 3.3)
    draw_box(ax, 2.6, 2.6, 1.8, 1.4, "Data pipeline",
             sublabel="Filter · encode · implicit feedback",
             facecolor="#EEEDFE", edgecolor="#534AB7",
             labelcolor="#26215C", sublabelcolor="#3C3489")
    draw_arrow(ax, 4.4, 3.3, 4.9, 3.3)
    draw_container(ax, 4.9, 2.35, 3.8, 3.2, "STAGE 1 — CANDIDATE GENERATION",
                   edgecolor="#0F6E56", facecolor="#F0FAF6")
    draw_box(ax, 5.1, 2.6, 1.6, 1.4, "ALS",
             sublabel="128-dim embeddings",
             facecolor="#E1F5EE", edgecolor="#0F6E56",
             labelcolor="#04342C", sublabelcolor="#085041")
    draw_arrow(ax, 6.7, 3.3, 7.1, 3.3)
    draw_box(ax, 7.1, 2.6, 1.6, 1.4, "FAISS",
             sublabel="Top 100 candidates",
             facecolor="#E1F5EE", edgecolor="#0F6E56",
             labelcolor="#04342C", sublabelcolor="#085041")
    draw_arrow(ax, 8.7, 3.3, 9.2, 3.3)
    draw_container(ax, 9.2, 2.35, 3.8, 3.2, "STAGE 2 — RANKING MODEL",
                   edgecolor="#993C1D", facecolor="#FEF6F3")
    draw_box(ax, 9.4, 2.6, 1.6, 1.4, "Features",
             sublabel="340-dim feature vec",
             facecolor="#FAECE7", edgecolor="#993C1D",
             labelcolor="#4A1B0C", sublabelcolor="#712B13")
    draw_arrow(ax, 11.0, 3.3, 11.4, 3.3)
    draw_box(ax, 11.4, 2.6, 1.6, 1.4, "MLP ranker",
             sublabel="256-128-64 · 92% accuracy",
             facecolor="#FAECE7", edgecolor="#993C1D",
             labelcolor="#4A1B0C", sublabelcolor="#712B13")
    draw_arrow(ax, 13.0, 3.3, 13.4, 3.3)
    draw_box(ax, 13.4, 2.6, 1.6, 1.4, "Top 10",
             sublabel="Ranked results",
             facecolor="#EEEDFE", edgecolor="#534AB7",
             labelcolor="#26215C", sublabelcolor="#3C3489")
    draw_box(ax, 13.4, 1.6, 1.6, 0.8, "Interaction logger",
             facecolor="#F5F5F5", edgecolor="#AAAAAA",
             labelcolor="#222222", fontsize=8.5)
    draw_arrow(ax, 14.2, 2.6, 14.2, 2.4)
    draw_container(ax, 0.2, 0.2, 11.0, 2.0, "COLD START — NEW USER HANDLING",
                   edgecolor="#854F0B", facecolor="#FFFBF5")
    draw_box(ax, 0.4, 0.45, 2.0, 1.4, "New user",
             sublabel="No interaction history",
             facecolor="#FAEEDA", edgecolor="#854F0B",
             labelcolor="#412402", sublabelcolor="#633806")
    draw_arrow(ax, 2.4, 1.15, 2.8, 1.15, color="#854F0B")
    draw_box(ax, 2.8, 0.45, 2.4, 1.4, "Popularity fallback",
             sublabel="Recency-weighted · genre filter",
             facecolor="#FAEEDA", edgecolor="#854F0B",
             labelcolor="#412402", sublabelcolor="#633806")
    draw_arrow(ax, 5.2, 1.15, 5.6, 1.15, color="#854F0B")
    draw_box(ax, 5.6, 0.45, 2.4, 1.4, "Content similarity",
             sublabel="Genome + genres · FAISS index",
             facecolor="#FAEEDA", edgecolor="#854F0B",
             labelcolor="#412402", sublabelcolor="#633806")
    draw_arrow(ax, 8.0, 1.15, 8.4, 1.15, color="#854F0B")
    draw_box(ax, 8.4, 0.45, 2.4, 1.4, "Top 10 popular",
             sublabel="Best available recommendations",
             facecolor="#FAEEDA", edgecolor="#854F0B",
             labelcolor="#412402", sublabelcolor="#633806")

    legend_x = 12.0
    legend_items = [
        ("#E1F5EE", "#0F6E56", "Stage 1"),
        ("#FAECE7", "#993C1D", "Stage 2"),
        ("#FAEEDA", "#854F0B", "Cold start"),
        ("#EEEDFE", "#534AB7", "System"),
    ]
    for i, (fc, ec, label) in enumerate(legend_items):
        rect = FancyBboxPatch((legend_x, 0.5 + i * 0.42), 0.3, 0.28,
                              boxstyle="round,pad=0.02",
                              facecolor=fc, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.45, 0.64 + i * 0.42, label,
                va="center", fontsize=9, color="#444444")

    ax.text(8.0, 0.05,
            "Precision@10: 0.308  ·  Recall@10: 0.069  ·  NDCG@10: 0.320  ·  Evaluated on 1000 users",
            ha="center", va="center", fontsize=8, color="#999999")

    plt.tight_layout(pad=0.2)
    fig.savefig(arch_path, dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def generate_pipeline_image(pipeline_path):
    fig, ax = plt.subplots(figsize=(16, 2.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    def draw_box(ax, x, y, w, h, label, sublabel=None, facecolor="#F5F5F5",
                 edgecolor="#AAAAAA", labelcolor="#222222", sublabelcolor="#555555"):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=facecolor,
                             edgecolor=edgecolor,
                             linewidth=1.2)
        ax.add_patch(box)
        if sublabel:
            ax.text(x + w / 2, y + h * 0.65, label,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=labelcolor)
            ax.text(x + w / 2, y + h * 0.28, sublabel,
                    ha="center", va="center",
                    fontsize=9, color=sublabelcolor)
        else:
            ax.text(x + w / 2, y + h / 2, label,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=labelcolor)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2))

    boxes = [
        ("Raw CSVs", "6 files · 20M rows",
         "#F5F5F5", "#AAAAAA", "#222222", "#555555"),
        ("Preprocessing", "Filter · encode · split",
         "#EEEDFE", "#534AB7", "#26215C", "#3C3489"),
        ("ALS + FAISS", "Embeddings · index",
         "#E1F5EE", "#0F6E56", "#04342C", "#085041"),
        ("MLP training", "PyTorch · MPS GPU",
         "#FAECE7", "#993C1D", "#4A1B0C", "#712B13"),
        ("Evaluation", "Precision · Recall · NDCG",
         "#EEEDFE", "#534AB7", "#26215C", "#3C3489"),
    ]

    box_w = 2.6
    gap = 0.5
    start_x = 0.4

    for i, (label, sublabel, fc, ec, lc, slc) in enumerate(boxes):
        x = start_x + i * (box_w + gap)
        draw_box(ax, x, 0.4, box_w, 1.4, label, sublabel, fc, ec, lc, slc)
        if i < len(boxes) - 1:
            draw_arrow(ax, x + box_w, 1.1, x + box_w + gap, 1.1)

    plt.tight_layout(pad=0.2)
    fig.savefig(pipeline_path, dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def tab_overview():
    config = load_config()
    model_dir = config["paths"]["models"]

    arch_path = os.path.join(model_dir, "architecture.png")
    pipeline_path = os.path.join(model_dir, "pipeline.png")

    if not os.path.exists(arch_path):
        generate_architecture_image(arch_path)

    if not os.path.exists(pipeline_path):
        generate_pipeline_image(pipeline_path)

    st.header("System Overview")
    st.write(
        "This application demonstrates an industry-level two-stage neural recommendation system "
        "trained on the MovieLens 20M dataset. "
        "The system narrows down 27,000 movies to a small set of candidates using learned embeddings, "
        "then ranks those candidates using a neural network that combines collaborative and content-based features. "
        "Cold start handling ensures the system can serve meaningful recommendations even for brand new users."
    )

    st.subheader("System Architecture")
    st.image(arch_path, use_column_width=True)

    st.subheader("Data Pipeline Workflow")
    st.image(pipeline_path, use_column_width=True)

    st.subheader("Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings", "20,000,263")
    col2.metric("Total Users", "138,493")
    col3.metric("Total Movies", "27,278")
    col4.metric("Train / Test Split", "80% / 20%")

    st.subheader("Tech Stack")
    tech_data = {
        "Component": [
            "Collaborative filtering",
            "Approximate nearest neighbor",
            "Neural ranking model",
            "Content features",
            "Experiment tracking",
            "Interaction logging",
            "Application framework",
            "GPU acceleration"
        ],
        "Tool": [
            "Implicit (ALS)",
            "FAISS",
            "PyTorch MLP",
            "Genome tag scores",
            "MLflow",
            "Supabase (PostgreSQL)",
            "Streamlit",
            "Apple MPS"
        ],
        "Purpose": [
            "Learn user and item embeddings from implicit feedback",
            "Fast top-K retrieval over 18K item vectors",
            "Re-rank candidates using rich feature combinations",
            "128 tag relevance scores per movie for content similarity",
            "Track training runs, hyperparameters, and metrics locally",
            "Log user interactions and recommendations persistently",
            "Interactive demo interface for the full pipeline",
            "GPU-accelerated training on Apple Silicon"
        ]
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)


def tab_existing_user(config, live_mode, user_embeddings, item_embeddings, faiss_index,
                      user_encoder, movie_encoder, ranking_model, train, movies, genome):
    st.header("Existing User Recommendations")
    st.write(
        "Select a user to run the full two stage recommendation pipeline. "
        "Stage 1 uses FAISS to retrieve 100 candidates from the embedding space. "
        "Stage 2 uses a trained neural network to rank those candidates and return the top 10 recommendations."
    )

    if live_mode:
        active_users = sample_active_users(train)
    else:
        precomputed = load_precomputed("sample_recommendations.json")
        active_users = [int(k) for k in precomputed.keys()]
        st.info(
            "Running in demo mode. Recommendations are precomputed from the trained model. "
            "All results are real outputs from the two-stage pipeline."
        )

    selected_user_idx = st.selectbox("Select a User ID", active_users)

    if live_mode:
        freshness_bonus = st.slider(
            "Freshness preference",
            min_value=0.0, max_value=0.5, value=0.0, step=0.05,
            help="0.0 ranks purely by relevance. Higher values nudge recommendations toward newer movies.",
            key="existing_user_freshness"
        )
        col1, col2, col3 = st.columns(3)
        col1.caption("0.0 — Pure relevance")
        col2.caption("0.25 — Balanced")
        col3.caption("0.5 — Prefer recent")
        genre_matrix, genome_features = prepare_content_features(movies, genome)

    if st.button("Get Recommendations"):
        if live_mode:
            st.subheader("Watch History")
            history = get_user_watch_history(selected_user_idx, train, movies)
            if len(history) > 0:
                render_movie_table(history)
            else:
                st.write("No watch history available for this user.")

            with st.spinner("Running candidate generation and ranking..."):
                try:
                    recommended_ids = retrieve_and_rank(
                        selected_user_idx, user_embeddings, faiss_index,
                        item_embeddings, ranking_model, movie_encoder,
                        genre_matrix, genome_features, config,
                        movies=movies, freshness_bonus=freshness_bonus
                    )
                    recommended_movies = movies[movies["movieId"].isin(recommended_ids)][
                        ["movieId", "title_clean", "genres", "year"]
                    ]
                    log_recommendations(
                        user_id=selected_user_idx,
                        movie_ids=recommended_ids.tolist(),
                        stage="two_stage"
                    )
                    for movie_id in recommended_ids:
                        log_interaction(
                            user_id=selected_user_idx,
                            movie_id=int(movie_id),
                            event_type="recommended"
                        )
                    freshness_label = (
                        "pure relevance" if freshness_bonus == 0.0
                        else f"freshness boost {freshness_bonus}"
                    )
                    st.subheader(f"Top 10 Recommendations ({freshness_label})")
                    render_movie_table(recommended_movies)
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
        else:
            user_data = precomputed.get(str(selected_user_idx), {})

            st.subheader("Watch History")
            history = user_data.get("watch_history", [])
            if history:
                history_df = pd.DataFrame(history)
                history_df.index += 1
                st.dataframe(history_df.rename(columns={
                    "title_clean": "Title", "genres": "Genres", "year": "Year"
                }), use_container_width=True)
            else:
                st.write("No watch history available for this user.")

            st.subheader("Top 10 Recommendations")
            recs = user_data.get("recommendations", [])
            if recs:
                recs_df = pd.DataFrame(recs)
                recs_df.index += 1
                st.dataframe(recs_df.rename(columns={
                    "title_clean": "Title", "genres": "Genres", "year": "Year"
                }), use_container_width=True)
                log_recommendations(
                    user_id=selected_user_idx,
                    movie_ids=[r.get("title_clean") for r in recs],
                    stage="precomputed"
                )
            else:
                st.write("No recommendations available for this user.")


def tab_cold_start(config, live_mode, movies):
    st.header("Cold Start Recommendations")
    st.write(
        "This tab demonstrates how the system handles a brand new user with no interaction history. "
        "Select one or more preferred genres and the system will return the most popular movies "
        "that match your preferences. If no genre is selected, the system returns globally popular movies."
    )

    if live_mode:
        genre_cols = get_genre_cols(movies)
        available_genres = [g for g in genre_cols if g != "(no genres listed)"]
        popular_df = None
    else:
        genre_data = load_precomputed("sample_genre_data.json")
        available_genres = genre_data["all_genres"]
        popular_raw = load_precomputed("sample_popular.json")
        popular_df = pd.DataFrame(popular_raw)

    selected_genres = st.multiselect("Select Preferred Genres (optional)", available_genres)

    freshness_bonus = st.slider(
        "Freshness preference",
        min_value=0.0, max_value=0.5, value=0.0, step=0.05,
        help="0.0 ranks purely by popularity. Higher values nudge recommendations toward newer movies.",
        key="cold_start_freshness"
    )
    col1, col2, col3 = st.columns(3)
    col1.caption("0.0 — Pure popularity")
    col2.caption("0.25 — Balanced")
    col3.caption("0.5 — Prefer recent")

    if st.button("Get Cold Start Recommendations"):
        with st.spinner("Fetching recommendations..."):
            if live_mode:
                from src.cold_start import recommend_for_new_user
                recommendations = recommend_for_new_user(
                    config,
                    genre_filter=selected_genres if selected_genres else None,
                    top_k=50
                )
            else:
                recommendations = popular_df.copy()
                if selected_genres:
                    mask = recommendations[selected_genres].any(axis=1) if all(
                        g in recommendations.columns for g in selected_genres
                    ) else pd.Series([True] * len(recommendations))
                    recommendations = recommendations[mask]

            if len(recommendations) > 0:
                if freshness_bonus > 0.0:
                    if live_mode:
                        movie_year_map = movies[["movieId", "year"]].dropna().set_index(
                            "movieId"
                        )["year"].to_dict()
                        years = recommendations["movieId"].map(movie_year_map)
                    else:
                        years = pd.to_numeric(recommendations["year"], errors="coerce")

                    min_year = years.min()
                    max_year = years.max()
                    year_range = max_year - min_year if max_year != min_year else 1.0
                    recency_weights = (years - min_year) / year_range
                    recency_weights = recency_weights.fillna(0.0)
                    max_score = recommendations["final_score"].max()
                    recommendations = recommendations.copy()
                    recommendations["final_score"] = (
                        recommendations["final_score"] / max_score +
                        freshness_bonus * recency_weights.values
                    )
                    recommendations = recommendations.sort_values(
                        "final_score", ascending=False
                    )

                recommendations = recommendations.head(10)
                freshness_label = (
                    "pure popularity" if freshness_bonus == 0.0
                    else f"freshness boost {freshness_bonus}"
                )
                st.subheader(f"Recommended Movies ({freshness_label})")
                render_movie_table(
                    recommendations,
                    score_col="final_score",
                    score_label="Score"
                )
            else:
                st.write("No recommendations found for the selected genres.")


def tab_similar_movies(config, live_mode, movies, movie_meta,
                       content_faiss_index, content_movie_ids, content_matrix):
    st.header("Similar Movies")
    st.write(
        "Select a movie to find the most content similar titles in the catalog. "
        "Similarity is computed using genre features and genome tag scores, "
        "indexed with FAISS for fast retrieval."
    )

    if live_mode:
        all_titles = movies[["movieId", "title_clean"]].dropna().drop_duplicates("movieId")
        title_to_id = dict(zip(all_titles["title_clean"], all_titles["movieId"]))
        sorted_titles = sorted(title_to_id.keys())
    else:
        similar_data = load_precomputed("sample_similar_movies.json")
        sorted_titles = sorted(similar_data.keys())

    selected_title = st.selectbox("Search for a Movie", sorted_titles)

    freshness_bonus = st.slider(
        "Freshness preference",
        min_value=0.0, max_value=0.5, value=0.0, step=0.05,
        help="0.0 ranks purely by content similarity. Higher values nudge results toward newer movies.",
        key="similar_movies_freshness"
    )
    col1, col2, col3 = st.columns(3)
    col1.caption("0.0 — Pure similarity")
    col2.caption("0.25 — Balanced")
    col3.caption("0.5 — Prefer recent")

    if st.button("Find Similar Movies"):
        with st.spinner("Searching content index..."):
            if live_mode:
                from src.cold_start import get_similar_movies
                movie_id = title_to_id.get(selected_title)
                if movie_id is None:
                    st.error("Movie not found in the catalog.")
                    return
                similar = get_similar_movies(movie_id, config, top_k=50)
                similar_display = similar.reset_index()
            else:
                results = similar_data.get(selected_title, [])
                if not results:
                    st.write("No similar movies found for the selected title.")
                    return
                similar_display = pd.DataFrame(results)

            if len(similar_display) > 0:
                if freshness_bonus > 0.0:
                    years = pd.to_numeric(similar_display["year"], errors="coerce")
                    min_year = years.min()
                    max_year = years.max()
                    year_range = max_year - min_year if max_year != min_year else 1.0
                    recency_weights = (years - min_year) / year_range
                    recency_weights = recency_weights.fillna(0.0)
                    max_sim = similar_display["similarity_score"].max()
                    similar_display = similar_display.copy()
                    similar_display["similarity_score"] = (
                        similar_display["similarity_score"] / max_sim +
                        freshness_bonus * recency_weights.values
                    )
                    similar_display = similar_display.sort_values(
                        "similarity_score", ascending=False
                    )

                similar_display = similar_display.head(10)
                freshness_label = (
                    "pure similarity" if freshness_bonus == 0.0
                    else f"freshness boost {freshness_bonus}"
                )
                st.subheader(f"Movies Similar to {selected_title} ({freshness_label})")
                render_movie_table(
                    similar_display,
                    score_col="similarity_score",
                    score_label="Score"
                )
            else:
                st.write("No similar movies found for the selected title.")


def tab_system_insights(config, live_mode, user_embeddings, item_embeddings,
                        faiss_index, popularity, movies):
    st.header("System Insights")

    st.subheader("System Statistics")
    col1, col2, col3, col4 = st.columns(4)

    if live_mode:
        col1.metric("Total Users", f"{user_embeddings.shape[0]:,}")
        col2.metric("Total Items", f"{item_embeddings.shape[0]:,}")
        col3.metric("Embedding Dimension", user_embeddings.shape[1])
        col4.metric("FAISS Index Size", f"{faiss_index.ntotal:,}")
    else:
        col1.metric("Total Users", "138,493")
        col2.metric("Total Items", "18,345")
        col3.metric("Embedding Dimension", "128")
        col4.metric("FAISS Index Size", "18,345")

    st.markdown("---")

    row1_col1, row1_col2 = st.columns(2)

    metrics_data = {
        "K": [5, 10, 20],
        "Precision": [0.3310, 0.3078, 0.1539],
        "Recall": [0.0370, 0.0690, 0.0690],
        "NDCG": [0.3381, 0.3204, 0.2113]
    }
    metrics_df = pd.DataFrame(metrics_data).set_index("K")

    with row1_col1:
        st.subheader("Evaluation Metrics")
        st.write(
            "Evaluated on 1000 held out users using a temporal train test split. "
            "K = 5 is the strictest setting. "
            "K = 10 is the standard industry benchmark. "
            "K = 20 measures broader coverage of user taste. "
            "NDCG penalizes relevant items ranked lower in the list."
        )
        st.dataframe(metrics_df, use_container_width=True)

    with row1_col2:
        st.subheader("Evaluation Chart")
        k_labels = ["K = 5", "K = 10", "K = 20"]
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            name="Precision", x=k_labels, y=metrics_data["Precision"],
            marker_color="#534AB7",
            text=[f"{v:.3f}" for v in metrics_data["Precision"]],
            textposition="outside"
        ))
        fig_metrics.add_trace(go.Bar(
            name="Recall", x=k_labels, y=metrics_data["Recall"],
            marker_color="#0F6E56",
            text=[f"{v:.3f}" for v in metrics_data["Recall"]],
            textposition="outside"
        ))
        fig_metrics.add_trace(go.Bar(
            name="NDCG", x=k_labels, y=metrics_data["NDCG"],
            marker_color="#993C1D",
            text=[f"{v:.3f}" for v in metrics_data["NDCG"]],
            textposition="outside"
        ))
        fig_metrics.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 0.5], title="Score"),
            xaxis=dict(title="K"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(t=40, b=40),
            height=380
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("---")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Distribution of Ratings")
        st.write(
            "Most users rate movies they enjoyed, creating a positive skew. "
            "Ratings of 3.5 and above are treated as positive implicit feedback signals."
        )

        if live_mode:
            processed = config["paths"]["processed_data"]
            train = pd.read_parquet(os.path.join(processed, "train.parquet"))
            rating_counts = train["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["Rating", "Count"]
        else:
            ratings_data = load_precomputed("sample_ratings.json")
            rating_counts = pd.DataFrame({
                "Rating": ratings_data["ratings"],
                "Count": ratings_data["counts"]
            })

        fig_ratings = px.bar(
            rating_counts, x="Rating", y="Count",
            color_discrete_sequence=["#534AB7"], text="Count"
        )
        fig_ratings.update_traces(
            texttemplate="%{text:.2s}", textposition="outside"
        )
        fig_ratings.add_vline(
            x=3.5, line_dash="dash", line_color="#993C1D",
            annotation_text="Implicit threshold (3.5)",
            annotation_position="top right"
        )
        fig_ratings.update_layout(
            xaxis=dict(title="Rating"),
            yaxis=dict(title="Number of ratings"),
            margin=dict(t=40, b=40), height=380
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    with row2_col2:
        st.subheader("Genre Co-occurrence")
        st.write(
            "Shows how often two genres appear together in the same movie. "
            "Drama and Comedy are the most frequently paired genres in the catalog."
        )

        if live_mode:
            processed = config["paths"]["processed_data"]
            movies_processed = pd.read_parquet(os.path.join(processed, "movies.parquet"))
            exclude = ["movieId", "title", "genres", "year", "title_clean"]
            genre_cols = [c for c in movies_processed.columns if c not in exclude]
            top_genres = movies_processed[genre_cols].sum().sort_values(
                ascending=False
            ).head(10).index.tolist()
            genre_matrix_data = movies_processed[top_genres].values
            cooccurrence = genre_matrix_data.T @ genre_matrix_data
            np.fill_diagonal(cooccurrence, 0)
            cooccurrence_df = pd.DataFrame(
                cooccurrence, index=top_genres, columns=top_genres
            )
        else:
            genre_data = load_precomputed("sample_genre_data.json")
            top_genres = genre_data["top_genres"]
            cooccurrence = np.array(genre_data["cooccurrence"])
            np.fill_diagonal(cooccurrence, 0)
            cooccurrence_df = pd.DataFrame(
                cooccurrence, index=top_genres, columns=top_genres
            )

        fig_heatmap = px.imshow(
            cooccurrence_df, color_continuous_scale="Blues",
            text_auto=".0f", aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis=dict(title=""),
            yaxis=dict(title=""),
            margin=dict(t=40, b=40),
            height=380,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    st.subheader("Top N Most Popular Movies")
    row3_chart, row3_controls = st.columns([3, 1])

    if live_mode:
        exclude = ["movieId", "title", "genres", "year", "title_clean"]
        genre_cols_all = [c for c in movies.columns if c not in exclude
                          and c != "(no genres listed)"]
        popular_source = popularity.copy()
    else:
        genre_data = load_precomputed("sample_genre_data.json")
        genre_cols_all = genre_data["all_genres"]
        popular_raw = load_precomputed("sample_popular.json")
        popular_source = pd.DataFrame(popular_raw)

    with row3_controls:
        st.write("Controls")
        n_movies = st.slider(
            "Number of movies", min_value=5, max_value=50, value=20, step=5
        )
        selected_genres = st.multiselect(
            "Filter by genre", options=genre_cols_all, default=[]
        )

    with row3_chart:
        filtered_popularity = popular_source.copy()

        if selected_genres:
            valid_cols = [g for g in selected_genres if g in filtered_popularity.columns]
            if valid_cols:
                genre_mask = filtered_popularity[valid_cols].any(axis=1)
                filtered_popularity = filtered_popularity[genre_mask]

        top_n = filtered_popularity.sort_values(
            "interaction_count", ascending=False
        ).head(n_movies)

        if len(top_n) == 0:
            st.write("No movies found for the selected genres.")
        else:
            fig_popular = px.bar(
                top_n[::-1].reset_index(drop=True),
                x="interaction_count", y="title_clean",
                orientation="h",
                color_discrete_sequence=["#534AB7"],
                text="interaction_count",
                labels={
                    "interaction_count": "Number of unique users",
                    "title_clean": ""
                }
            )
            fig_popular.update_traces(
                texttemplate="%{text:,}", textposition="outside"
            )
            fig_popular.update_layout(
                margin=dict(t=20, b=40, l=20, r=120),
                height=max(400, n_movies * 28),
                yaxis=dict(automargin=True)
            )
            st.plotly_chart(fig_popular, use_container_width=True)

    st.markdown("---")

    st.subheader("Recent Interaction Logs")
    st.write("The following interactions were logged during this session.")
    try:
        recent = get_recent_interactions(limit=20)
        if recent:
            logs_df = pd.DataFrame(recent)
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.write(
                "No interactions logged yet. "
                "Generate recommendations in the other tabs to see logs here."
            )
    except Exception as e:
        st.write(f"Could not fetch interaction logs: {e}")


def main():
    st.set_page_config(
        page_title="Neural Recommendation Engine",
        layout="wide"
    )

    st.title("Neural Recommendation Engine")
    st.write(
        "This application demonstrates a two stage neural recommendation system "
        "trained on the MovieLens 20M dataset. "
        "Stage 1 uses Alternating Least Squares embeddings with FAISS for candidate generation. "
        "Stage 2 uses a multilayer perceptron to rank candidates using collaborative and content features."
    )

    with st.spinner("Loading models and data..."):
        artifacts = load_all_artifacts()

    (config, live_mode, user_embeddings, item_embeddings, faiss_index,
     user_encoder, movie_encoder, ranking_model, train, movies,
     genome, popularity, movie_meta, content_faiss_index,
     content_movie_ids, content_matrix) = artifacts

    if live_mode:
        st.success("Running in full mode with live model inference.")
    else:
        st.info(
            "Running in demo mode with precomputed results. "
            "All recommendations are real outputs from the trained two-stage pipeline."
        )

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "System Overview",
        "Existing User",
        "Cold Start",
        "Similar Movies",
        "System Insights"
    ])

    with tab0:
        tab_overview()

    with tab1:
        tab_existing_user(
            config, live_mode, user_embeddings, item_embeddings, faiss_index,
            user_encoder, movie_encoder, ranking_model, train, movies, genome
        )

    with tab2:
        tab_cold_start(config, live_mode, movies)

    with tab3:
        tab_similar_movies(
            config, live_mode, movies, movie_meta,
            content_faiss_index, content_movie_ids, content_matrix
        )

    with tab4:
        tab_system_insights(
            config, live_mode, user_embeddings, item_embeddings,
            faiss_index, popularity, movies
        )


if __name__ == "__main__":
    main()