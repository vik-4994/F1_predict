# f1-rank

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_features.py --config configs/base.yaml
python scripts/train_ranker.py --config configs/model/lgbm_ranker.yaml
python scripts/evaluate.py --run latest
python scripts/predict_race.py --run latest --race-id 1100
```

## Structure
- `src/f1rank/*` — libraries (modules)
- `scripts/*` — CLI-scripts
- `configs/*` — configs
- `data/{raw,interim,features,artifacts}` — data & artifacts
- `api/app.py` — FastAPI-service
- `ui/streamlit_app.py` — UI

## Additional info
- Main target: `finish_pos = results.positionOrder` (filter DNS/DNQ; DSQ — setting).