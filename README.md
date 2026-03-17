# f1-rank

An ML project for pre-race Formula 1 finishing position prediction.
The repository includes:

- raw FastF1 data export and incremental refresh;
- pre-race feature and target building;
- PyTorch listwise/ranking model training;
- CLI prediction for a specific weekend and season simulation;
- a FastAPI backend and React dashboard for local operations and season projections;
- a legacy Streamlit UI for interactive inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full React + API stack with Docker:

```bash
docker compose up --build
```

Then open:

- `http://localhost:3000` for the dashboard
- `http://localhost:8000/api/health` for the API health check

Quick check:

```bash
pytest -q tests/test_smoke.py
```

## Typical workflows

Refresh the latest completed FastF1 rounds and rebuild features:

```bash
python scripts/refresh_fastf1_pipeline.py --progress --with-targets --audit
```

Export raw CSV files directly from FastF1:

```bash
python scripts/export_last_two_years.py --completed-only --lookback-rounds 2
```

Rebuild features and targets from already exported raw CSV files:

```bash
python scripts/build_features.py --raw-dir data/raw_csv --out-dir data/processed --with-targets --progress
python scripts/audit_features.py --features data/processed/all_features.parquet
```

Train the main model:

```bash
python scripts/train.py --features data/processed/all_features.parquet --targets data/processed/all_targets.parquet --run-name baseline_v4
```

Train a dedicated future-only model for priors-only scenarios:

```bash
python scripts/train.py --features data/processed/all_features.parquet --targets data/processed/all_targets.parquet --run-name baseline_future_v1 --feature-profile future
```

Run race prediction:

```bash
python scripts/predict.py --artifacts models/baseline_v4 --raw-dir data/raw_csv --drivers VER,NOR,LEC --track "Australian Grand Prix" --sim-year 2025 --sim-round 1
```

Run prediction in `future` mode with a dedicated artifact:

```bash
python scripts/predict.py --artifacts models/baseline_v4 --future-artifacts models/baseline_future_v1 --raw-dir data/raw_csv --scenario-mode future --sim-year 2026 --sim-round 2
```

Launch the UI:

```bash
streamlit run ui/streamlit_app.py
```

Run the API backend:

```bash
uvicorn api.app:app --reload
```

Run the React dashboard:

```bash
cd web
npm install
npm run dev
```

Run the Dockerized dashboard and API:

```bash
docker compose up --build
```

## Current baseline

The current baseline defaults are defined in `src/training/config.py`:

- `hidden=128,64`
- `dropout=0.25`
- `lr=0.0007`
- `weight_decay=0.0003`
- `epochs=16`
- noisy strategy/pit features are excluded by default via ablation filters

`models/baseline_v4` is the current default artifact for inference and the UI.

## Project layout

- `src/` - core project library
- `src/features/` - pre-race feature generators
- `src/training/` - dataset, scaler, inference, and training engine
- `scripts/` - CLI entry points
- `api/` - FastAPI backend for the dashboard
- `web/` - React/Vite frontend dashboard
- `tests/` - smoke and regression tests
- `ui/streamlit_app.py` - legacy Streamlit UI
- `data/raw_csv/` - exported raw CSV files from FastF1
- `data/processed/` - processed parquet/csv feature and target tables
- `models/` - saved training artifacts
- `out/` - prediction and simulation outputs
- `reports/` - feature quality reports

## Notes

- The training target is multi-task: ranking among classified finishers plus a separate `finish / DNF / DSQ` outcome head.
- Export and refresh workflows require `fastf1` to be installed.
- The prediction pipeline supports both `observed` and `future` modes via `src/scenario_builder.py`.
- For `future` mode, a separate artifact trained with `--feature-profile future` is recommended.
- The React dashboard expects the API on `http://127.0.0.1:8000` and proxies `/api` requests during local dev.
- The Docker stack publishes the dashboard on `http://localhost:3000` and the API on `http://localhost:8000`.
- `docker-compose.yml` bind-mounts `data/`, `models/`, `out/`, and `reports/` so background jobs update host files directly.
