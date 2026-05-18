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

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/kbalagam/NN-movie-recommendation-engine.git
cd NN-movie-recommendation-engine
```

**2. Create and activate a virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download the MovieLens 20M dataset from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) and place the CSV files in `data/raw/`.

**5. Configure environment variables**

Create a `.env` file in the project root:

**6. Run the pipeline**
```bash
python src/data_pipeline.py
python src/candidate_generation.py
python src/ranking_model.py
python src/cold_start.py
python src/evaluation.py
```

**7. Launch the app**
```bash
streamlit run app/streamlit_app.py
```

## Key Design Decisions

**Why ALS over SVD?**
Pure SVD requires a dense matrix. With 138,000 users and 27,000 movies that is 3.7 billion entries. ALS works directly on the sparse interaction matrix and is significantly faster for implicit feedback data.

**Why FAISS for retrieval?**
Brute force nearest neighbor search over 27,000 item vectors is feasible at this scale but FAISS scales to millions of items without changing the serving code. Using IVF indexing with 100 clusters and 10 probes gives approximately 95% of brute force accuracy at a fraction of the computation.

**Why binary cross entropy loss for ranking?**
We frame ranking as binary classification — did the user interact with this item or not. BCE with negative sampling approximates pairwise ranking loss (BPR) in practice and is simpler to implement and debug.

**Why temporal train/test split?**
A random split causes data leakage — training on interactions that happened after test interactions. Temporal split ensures we always train on the past and evaluate on the future, which mirrors production behavior.

**Why a freshness boost instead of retraining?**
The dataset covers interactions up to 2016. Older movies have had more time to accumulate interactions, creating a popularity bias toward classic films. A configurable post-ranking freshness boost corrects this without retraining, and gives users control over the recency versus relevance trade-off.

## Challenges and How They Were Overcome

**Memory constraints during training**
Training the MLP ranker on all 9.4 million positive interactions with 4 negative samples each produced 47 million training samples — far exceeding the 16GB RAM available on the development machine. The solution was to sample 500,000 positive interactions (still a large and representative subset) and use vectorized NumPy operations instead of row-by-row Python iteration. This reduced sample building time from over 2 hours to under 5 minutes and kept memory usage within safe limits. The ALS model still trained on all 20 million interactions, so the embedding quality was not compromised.

**FAISS and PyTorch conflict on Apple Silicon**
Loading FAISS and PyTorch in the same process caused segmentation faults on Apple Silicon due to conflicting OpenMP thread libraries. The fix was to set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` as environment variables before any imports. All inference in the Streamlit app and evaluation script runs on CPU to avoid the MPS conflict entirely, while training uses MPS for GPU acceleration.

**Timestamp format inconsistency**
The raw ratings file stores timestamps as Unix integers but after processing through the data pipeline they were saved as datetime strings in parquet format. This caused the cold start recency weighting to fail silently — `pd.to_numeric` returned all NaN values. The fix was to use `pd.to_datetime` with `dt.total_seconds()` for the time range calculation, producing correct recency weights.

**Popularity list merge failure**
The cold start popularity list initially returned an empty dataframe despite correct data in both the train set and movies table. The root cause was that the recency weight calculation was failing before the merge, producing an empty dataframe that masked the real data. Adding explicit debug print statements at each stage isolated the issue to the timestamp conversion, which was then fixed as described above.

**Streamlit Cloud deployment constraints**
The trained models and processed data total over 1GB and cannot be stored in the GitHub repository. The solution was a two-mode architecture: the app detects at startup whether model files are present and switches between live inference mode (local) and precomputed demo mode (cloud). A dedicated precomputation script generates six small JSON files (2.9MB total) covering recommendations for 20 users, popular movies, genre data, rating distributions, and content similarity results. These are committed to the repository and power the cloud deployment without any model files.

## Current Limitations

**Dataset recency**
The MovieLens 20M dataset covers interactions up to 2016. The model has no knowledge of movies released after 2016 and cannot recommend them through the collaborative filtering pipeline. The freshness boost partially addresses this for the popularity and content-based components but the core ALS embeddings remain bounded by the training data.

**Popularity bias**
Movies with more historical interactions produce stronger ALS embeddings, causing the system to favor well-established older films over newer releases. The configurable freshness boost is a post-ranking correction rather than a model-level solution.

**Cold start coverage**
The content similarity index covers only the 10,345 movies that have genome tag scores. Movies without genome data fall back to genre-only similarity, which is less precise.

**No real-time retraining**
The system logs user interactions to Supabase but does not use them to update the model. In a production system, logged interactions would feed into a periodic retraining pipeline to keep recommendations fresh.

**Fixed negative sampling**
During MLP training, negatives are sampled uniformly at random from the full item catalog. More sophisticated strategies like popularity-weighted negative sampling or hard negative mining would improve ranking quality, particularly for items in the long tail.

## Future Scope

**Add release year as a ranking feature**
Including normalized release year in the MLP feature vector would allow the model to learn personalized recency preferences from data — some users prefer classic films while others prefer recent releases. This is a more principled solution than the current global freshness boost.

**Implement BPR loss**
Replacing binary cross entropy with Bayesian Personalized Ranking loss would directly optimize the ranking order rather than binary relevance, which is theoretically more aligned with the recommendation objective.

**Session-based recommendations**
Adding a sequential model (GRU or transformer) on top of the two-stage pipeline would allow the system to incorporate within-session context — what the user has clicked on in the current session — for more dynamic recommendations.

**Online retraining pipeline**
Building an Airflow or Prefect pipeline that periodically retrains the ALS model on fresh interaction logs from Supabase would keep recommendations current without manual intervention.

**A/B testing framework**
Implementing a simple A/B testing layer that routes users to different model variants and compares their interaction rates would enable data-driven model selection in production.

**Expand to multi-modal features**
Incorporating movie poster embeddings (via a pretrained CNN) and plot synopsis embeddings (via a sentence transformer) alongside the existing genome and genre features would enrich the ranking model and improve cold start coverage for movies without genome scores.

## Streamlit Application

The application has five tabs:

- **System Overview** — architecture diagram, data pipeline workflow, dataset statistics, and tech stack
- **Existing User** — full two-stage pipeline with configurable freshness preference
- **Cold Start** — popularity-based recommendations for new users with genre filtering and freshness preference
- **Similar Movies** — content-based similarity search using genome features and FAISS with freshness preference
- **System Insights** — evaluation metrics, charts, top popular movies, and real-time interaction logs

## Author

Keerthan Balagam
