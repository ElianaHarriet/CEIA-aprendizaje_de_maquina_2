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

## In Progress 🔄

- [ ] Wait for training DAG to complete (Optuna 50 trials running)
- [ ] Verify model registration in MLflow
- [ ] Restart FastAPI after model registration
- [ ] Test end-to-end prediction

## Pending 📋

- [ ] Remove unused DAGs (`asset_producer_dag.py`, `manual_test_parametrization.py`, `test_dag.py`)
- [ ] Add `.dockerignore` to repo root
- [ ] Clean up local `modelo/src/mlruns/` (optional, keep for reference)
- [ ] Verify DAG dependency: ETL must run before Training