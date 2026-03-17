# f1-rank

An ML project for pre-race Formula 1 race outcome prediction.
The repository includes:

- raw FastF1 data export and incremental refresh;
- pre-race feature and target building;
- PyTorch multi-task training for race ranking and outcome classification;
- CLI prediction for a specific weekend and season simulation;
- a FastAPI backend and React dashboard for local operations and season projections;
- a legacy Streamlit UI for interactive inference.

## Release highlights

- Multi-task race modeling: separate `finish / DNF / DSQ` outcome head plus ranking among classified finishers.
- Era-aware training weights: seasons from `2000+` are grouped into regulation eras and down/up-weighted during training.
- Season simulation supports `observed` and `future` scenario modes with separate artifacts.
- CLI/UI inference now exposes outcome probabilities such as `p_finish`, `p_dnf`, and `p_dsq`.

## Setup

```bash
python3 -m venv .venv
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
python3 -m pytest -q tests/test_smoke.py
```

## Typical workflows

Refresh the latest completed FastF1 rounds and rebuild features:

```bash
python3 scripts/refresh_fastf1_pipeline.py --progress --with-targets --audit
```

Export raw CSV files directly from FastF1:

```bash
python3 scripts/export_last_two_years.py --completed-only --lookback-rounds 2
```

Rebuild features and targets from already exported raw CSV files:

```bash
python3 scripts/build_features.py --raw-dir data/raw_csv --out-dir data/processed --with-targets --progress
python3 scripts/audit_features.py --features data/processed/all_features.parquet
```

Train the main multi-task model:

```bash
python3 scripts/train.py \
  --features data/processed/all_features.parquet \
  --targets data/processed/all_targets.parquet \
  --run-name baseline_v6 \
  --hidden 64,32 \
  --epochs 8 \
  --train-recency-half-life 6
```

Train a dedicated future-only model for priors-only scenarios:

```bash
python3 scripts/train.py \
  --features data/processed/all_features.parquet \
  --targets data/processed/all_targets.parquet \
  --run-name baseline_future_v2 \
  --feature-profile future
```

Run race prediction:

```bash
python3 scripts/predict.py \
  --artifacts models/baseline_v4 \
  --raw-dir data/raw_csv \
  --drivers VER,NOR,LEC \
  --track "Australian Grand Prix" \
  --sim-year 2025 \
  --sim-round 1
```

Run prediction in `future` mode with a dedicated artifact:

```bash
python3 scripts/predict.py \
  --artifacts models/baseline_v4 \
  --future-artifacts models/baseline_future_v2 \
  --raw-dir data/raw_csv \
  --scenario-mode future \
  --sim-year 2026 \
  --sim-round 2
```

Run full season simulation:

```bash
python3 scripts/simulate_season.py \
  --artifacts models/baseline_v4 \
  --future-artifacts models/baseline_future_v2 \
  --raw-dir data/raw_csv \
  --year 2026 \
  --scenario-mode future \
  --out-dir out/season_2026_future
```

Run the background job runner used by the UI:

```bash
python3 scripts/ui_job_runner.py \
  --job full_future \
  --year 2026 \
  --baseline-artifacts models/baseline_v4 \
  --future-run-name baseline_future_v2 \
  --scenario-mode future
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
- `status_loss_weight=1.0`
- noisy strategy/pit features are excluded by default via ablation filters
- regulation-era weights are enabled by default

Default regulation-era weights:

- `2000-2004` `v10_groove` = `0.08`
- `2005-2008` `v8_refuel` = `0.12`
- `2009-2013` `aero_kers` = `0.18`
- `2014-2016` `hybrid_v6_initial` = `0.30`
- `2017-2021` `wide_aero_hybrid` = `0.50`
- `2022-2025` `ground_effect` = `0.75`
- `2026+` `next_gen_2026` = `1.00`

`models/baseline_v4` is the current default artifact for inference and the UI.

Example custom era override:

```bash
python3 scripts/train.py \
  --features data/processed/all_features.parquet \
  --targets data/processed/all_targets.parquet \
  --run-name baseline_v6 \
  --era-weights "next_gen_2026=1.0,ground_effect=0.7,wide_aero_hybrid=0.45,hybrid_v6_initial=0.25,aero_kers=0.12,v8_refuel=0.08,v10_groove=0.05"
```

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
- The saved checkpoint is selected by a composite validation score based on finish-order quality and outcome quality.
- Training applies race recency weighting and, by default, regulation-era weighting.
- Export and refresh workflows require `fastf1` to be installed.
- The prediction pipeline supports both `observed` and `future` modes via `src/scenario_builder.py`.
- For `future` mode, a separate artifact trained with `--feature-profile future` is recommended.
- `predict.py` and the UI can expose `p_finish`, `p_dnf`, `p_dsq`, `predicted_outcome`, and `p_win`.
- The React dashboard expects the API on `http://127.0.0.1:8000` and proxies `/api` requests during local dev.
- The Docker stack publishes the dashboard on `http://localhost:3000` and the API on `http://localhost:8000`.
- `docker-compose.yml` bind-mounts `data/`, `models/`, `out/`, and `reports/` so background jobs update host files directly.
