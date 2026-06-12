# End-to-End ML Pipeline Documentation

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MinIO     │     │   Airflow   │     │   MLflow    │     │  FastAPI    │
│  (S3/Data)  │────▶│  (Orchestr.)│────▶│  (Tracking) │────▶│  (Serving)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  Raw/Interim/       ETL DAG            Model Registry       POST /predict
  Final buckets      Train DAG          Versioning           Health check
```

## DAG Execution Order

### Execution Order
```
etl_movielens → train_movielens → retrain_movielens (optional, repeatable)
```

### 1. ETL DAG (`etl_movielens`)
**Trigger**: Manual (no schedule)
**Tasks** (sequential):
1. `download_data` - Downloads MovieLens 25M from GroupLens, uploads raw CSVs to `s3://data/raw/`
2. `sample_and_save_ratings` - Samples users/ratings, saves to `s3://data/interim/ratings_sampled.csv`
3. `compute_movie_features` - Computes movie stats + genres + year → `s3://data/interim/movie_features.csv`
4. `compute_genome_pca` - PCA on genome tags (1200→20) → `s3://data/interim/genome_pca.csv`
5. `compute_user_features` - Computes user stats → `s3://data/interim/user_features.csv`
6. `merge_and_split` - Merges all features, creates interaction features, splits train/test → `s3://data/final/`

**Output**: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `feature_names.txt`

### 2. Training DAG (`train_movielens`)
**Trigger**: Manual (after ETL completes)
**Tasks**:
1. `train_and_register_model` - Loads splits from MinIO, runs Optuna (50 trials), trains XGBoost, calibrates, registers model in MLflow

**Output**: Registered model `movielens-rating-classifier` in MLflow

### 3. Retrain DAG (`retrain_movielens`)
**Trigger**: Manual (after Training completes)
**Tasks**:
1. `train_challenger` - Loads champion from MLflow, runs Optuna (5 trials centered on champion params), trains XGBoost, calibrates, registers as `challenger`
2. `evaluate_and_promote` - Loads champion + challenger from MLflow, compares F1 on test set, promotes winner to `champion` alias

**Champion/Challenger Flow**:
```
                    ┌──────────────────┐
                    │  train_movielens │
                    │  (initial model) │
                    └────────┬─────────┘
                             │ champion alias
                             v
                    ┌───────────────────┐
            ┌──────▶│ retrain_movielens │
            │       └─────────┬─────────┘
            │                 │
            │     ┌───────────┴───────────┐
            │     │                       │
            │     v                       v
            │  train_challenger    evaluate_and_promote
            │     │                       │
            │     │ challenger alias      │
            │     │                       │
            │     └───────────┬───────────┘
            │                 │
            │          ┌──────┴──────┐
            │          │             │
            │     challenger      champion
            │     F1 > champ?     F1 >= challenger?
            │          │             │
            │          ▼             ▼
            │     Promote to     Demote
            │     champion       challenger
            │          │
            └──────────┘
           (next retrain cycle)
```

**Output**: Updated alias on `movielens-rating-classifier` in MLflow registry

## Required Airflow Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `N_USERS` | 20000 | Number of users to sample |
| `N_RATINGS` | 1000000 | Max ratings after sampling |
| `RANDOM_STATE` | 42 | Random seed for reproducibility |
| `N_GENOME_COMPONENTS` | 20 | PCA components for genome tags |
| `TEST_SIZE` | 0.2 | Test split ratio |
| `TMP_DIR` | /opt/airflow/tmp/movielens | Temp directory for downloads |
| `ZIP_PATH` | /opt/airflow/tmp/ml-25m.zip | Path to downloaded zip |
| `RAW_PREFIX` | data/raw | MinIO prefix for raw data |
| `INTERIM_PREFIX` | data/interim | MinIO prefix for intermediate data |
| `FINAL_PREFIX` | final | MinIO prefix for final splits |
| `DATA_BUCKET` | data | MinIO bucket name |
| `MLFLOW_URI` | http://mlflow:5000 | MLflow tracking server URI |
| `EXPERIMENT_NAME` | movielens-rating-prediction | MLflow experiment name |
| `REGISTERED_MODEL_NAME` | movielens-rating-classifier | MLflow registered model name |

## MinIO Bucket Structure

```
s3://data/
├── raw/
│   ├── ratings.csv
│   ├── movies.csv
│   └── genome-scores.csv
├── interim/
│   ├── ratings_sampled.csv
│   ├── movie_features.csv
│   ├── genome_pca.csv
│   └── user_features.csv
└── final/
    ├── X_train.npy
    ├── X_test.npy
    ├── y_train.npy
    ├── y_test.npy
    └── feature_names.txt

s3://mlflow/
└── <experiment_id>/
    └── <run_id>/
        └── artifacts/
            ├── model/
            └── feature_names.txt
```

## FastAPI Endpoints

### POST /predict
**Request**:
```json
{
  "features": [0.1, 0.2, ... 50 values ...]
}
```

**Response**:
```json
{
  "probability": 0.73,
  "prediction": 1
}
```

### GET /health
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_uri": "models:/movielens-rating-classifier/latest"
}
```

### GET /docs
Interactive API documentation (Swagger UI)

## Feature Order (50 features)

Indices 0-19: Genre binary features (Action, Adventure, ..., (no genres listed))
Indices 20-23: Movie features (movie_avg_rating, movie_rating_count_log, movie_rating_std, year)
Indices 24-43: Genome PCA (genome_pca_0 through genome_pca_19)
Indices 44-47: User features (user_avg_rating, user_avg_rating_centered, user_rating_count_log, user_rating_std)
Indices 48-49: Interaction features (genre_cosine_similarity, user_deviation_from_movie_avg)

## Running the Pipeline

```bash
# 1. Start all services
docker compose --profile all up -d

# 2. Wait for all services to be healthy
docker ps --format "table {{.Names}}\t{{.Status}}"

# 3. Trigger ETL DAG in Airflow UI (http://localhost:8080)
#    Or via CLI:
docker exec <airflow-scheduler> airflow dags trigger etl_movielens

# 4. Wait for ETL to complete, then trigger Training DAG
docker exec <airflow-scheduler> airflow dags trigger train_movielens

# 5. (Optional) Trigger Retrain DAG to compare champion vs challenger
docker exec <airflow-scheduler> airflow dags trigger retrain_movielens

# 6. Verify model in MLflow UI (http://localhost:5001)

# 6. Test prediction
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'
```

## Troubleshooting

- **Airflow DAG not visible**: Check dag-processor logs, ensure DAG file has no syntax errors
- **Task fails with "Variable not found"**: Ensure variables are imported (run airflow-init or restart scheduler)
- **MinIO connection errors**: Check AWS_ENDPOINT_URL points to http://s3:9000
- **MLflow model not found**: Ensure training DAG completed successfully and model was registered
- **FastAPI 503 Model not loaded**: Restart FastAPI after model registration: `docker restart fastapi`