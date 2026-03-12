# f1-rank

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_features.py --raw-dir data/raw_csv --out-dir data/processed --with-targets --progress
python scripts/train.py --features data/processed/all_features.parquet --targets data/processed/all_targets.parquet --run-name baseline_v2
python scripts/predict.py --artifacts models/baseline_v2 --raw-dir data/raw_csv --drivers VER,NOR,LEC --track "Australian Grand Prix" --sim-year 2025 --sim-round 1
```

## Current baseline
- `train.py` defaults are tuned to the current best baseline found in this repo:
  - `hidden=128,64`
  - `dropout=0.25`
  - `lr=0.0007`
  - `weight_decay=0.0003`
  - `epochs=16`
  - strategy/pit noise is excluded by default via ablation filters

## Structure
- `src/f1rank/*` — libraries (modules)
- `scripts/*` — CLI-scripts
- `configs/*` — configs
- `data/{raw,interim,features,artifacts}` — data & artifacts
- `api/app.py` — FastAPI-service
- `ui/streamlit_app.py` — UI

## Additional info
- Main target: `finish_pos = results.positionOrder` (filter DNS/DNQ; DSQ — setting).
