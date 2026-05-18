# Neural Recommendation Engine

A two-stage neural recommendation system trained on the MovieLens 20M dataset. The system combines collaborative filtering, approximate nearest neighbor search, and a neural ranking model to deliver personalized movie recommendations at scale.

## Architecture

The system follows the two-stage architecture used in production recommendation systems at companies like Google, Netflix, and Spotify.

**Stage 1 — Candidate Generation**
- Trains an Alternating Least Squares model on 20 million implicit feedback interactions
- Learns 128-dimensional user and item embeddings
- Builds a FAISS IVF index over item embeddings for fast approximate nearest neighbor search
- Retrieves the top 100 candidate movies per user in milliseconds

**Stage 2 — Ranking Model**
- Takes the 100 candidates from Stage 1
- Constructs a 340-dimensional feature vector per candidate: user embedding + item embedding + genre features + genome tag scores
- Passes features through a PyTorch MLP (256-128-64) trained with binary cross entropy loss
- Re-ranks candidates by predicted relevance score and returns the top 10

**Cold Start Handling**
- New users with no interaction history receive popularity-based recommendations weighted by recency and interaction count
- New items are surfaced via content similarity using genome tag scores and genre features indexed with FAISS

## Dataset

MovieLens 20M — collected by GroupLens Research

| File | Description |
|---|---|
| rating.csv | 20 million ratings from 138,493 users on 27,278 movies |
| movie.csv | Movie titles and genres |
| genome_scores.csv | Tag relevance scores for 1,128 tags across 10,381 movies |
| genome_tags.csv | Tag names |
| tag.csv | User-generated tags |
| link.csv | Links to TMDb and IMDb |

## Results

Evaluated on 1,000 held-out users using a temporal train/test split (80/20).

| Metric | K=5 | K=10 | K=20 |
|---|---|---|---|
| Precision | 0.3310 | 0.3078 | 0.1539 |
| Recall | 0.0370 | 0.0690 | 0.0690 |
| NDCG | 0.3381 | 0.3204 | 0.2113 |

Precision@10 of 0.308 means that on average 3 out of every 10 recommendations are movies the user actually interacted with — out of a catalog of 27,000 movies.

## Tech Stack

| Component | Tool |
|---|---|
| Collaborative filtering | Implicit (ALS) |
| Approximate nearest neighbor | FAISS |
| Neural ranking model | PyTorch MLP |
| Content features | MovieLens genome tag scores |
| Experiment tracking | MLflow |
| Interaction logging | Supabase (PostgreSQL) |
| Application | Streamlit |
| GPU acceleration | Apple MPS |
