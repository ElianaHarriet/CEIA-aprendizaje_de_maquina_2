# TODO

## Completed ✅

- [x] Implement `POST /predict` endpoint in `dockerfiles/fastapi/app.py` - loads model from MLflow, returns probability
- [x] Add `GET /health` endpoint for service monitoring
- [x] Add Pydantic request/response validation with 50 features
- [x] Update `dockerfiles/fastapi/requirements.txt` with ML dependencies
- [x] Update `dockerfiles/fastapi/Dockerfile` with proper CMD
- [x] Add default values to Airflow variables in training DAG
- [x] Add `INTERIM_PREFIX` to Airflow variables
- [x] Document pipeline execution order, variables, and MinIO structure in `docs/PIPELINE.md`
- [x] Reduce Optuna trials to 10 and limit n_jobs=1 to prevent OOM
- [x] Training DAG completed successfully (F1=0.7367, AUC=0.8087)
- [x] Model registered in MLflow as `movielens-rating-classifier` v1
- [x] FastAPI loads model and serves predictions
- [x] All tests passing (10/10)

## Pending 📋

- [ ] Add `.dockerignore` to repo root (already done, verify in git)
- [ ] Clean up local `modelo/src/mlruns/` (optional, keep for reference)
- [ ] Verify DAG dependency: ETL must run before Training