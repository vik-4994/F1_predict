# `dev_trend_pre` — Pre-race Development Trends (Feature Doc)

Short, practical docs for the features produced by `dev_trend_pre.py`. One row per **current** driver; all numeric outputs are `float32`.

---

## Output columns

### `driver_trend`
- **Meaning:** Linear trend (slope) of the driver’s *race pace* over the last **K** past grands prix.
- **Sign:** **Higher = improving** (pace getting better).
- **Typical range:** ~`[-0.30, +0.30]` z-units per race (often ±0.10).

### `team_dev_trend`
- **Meaning:** Linear trend (slope) of the **team’s** average race pace over the last **K** past grands prix; broadcast to current team drivers.
- **Sign:** **Higher = improving** team pace.
- **Typical range:** ~`[-0.25, +0.25]` z-units per race.

### `stability_delta_vs_tm`
- **Meaning:** Robust variability of `(driver pace − team mean pace)` over the last **W** past grands prix, using `IQR/1.35` (≈σ for normal).
- **Sign:** **Lower = more stable** relative to team.
- **Typical range:** `[0.00, ~1.50]` (≤0.3 very stable; ≥0.9 volatile).

> Tiny: If your ranking interprets “higher score = better”, consider **inverting** `stability_delta_vs_tm` downstream or scaling it so that higher = better.

---

## Inputs (expected CSVs)

- `races.csv` – calendar (`year`, `round`).
- `laps_{Y}_{R}.csv` – lap times per driver (prefer a `milliseconds` column; if only `LapTime` exists, parse to ms).
- `pit_stops_{Y}_{R}.csv` *(optional)* – to drop in/out-laps (`Driver`, `lap`).
- `results_{Y}_{R}.csv` or `entrylist_{Y}_{R}_{Q|R}.csv` – driver → team mapping for that event.

---

## Context / parameters

- `dev_trend_window` = **K** (default **6**) — lookback for trend features.
- `stability_window` = **W** (default **8**) — lookback for stability.
- Current point in time is given by `year`, `round` in `ctx`.

---

## Computation pipeline (as-of, leak-safe)

1. **History selection (strictly before `year, round`):**  
   Use `races.csv` if available: keep `(y < year) or (y == year and r < round)`. Otherwise infer from filenames.

2. **Per-race pace for each past GP:**  
   - Load `laps_{y}_{r}.csv`. Work in milliseconds.  
   - Drop **in/out-laps** using `pit_stops` if present (`lap` and `lap+1`).  
   - For each driver: trimmed median lap (clip to [5,95] percentiles, then median).  
   - Convert to **pace z-score** over the field:  
     `pace_z = -(median_ms - mean_ms) / std_ms` (faster laps → larger `pace_z`).

3. **Team mean pace (per race):**  
   Map drivers to team for that event and average `pace_z` within team.

4. **`driver_trend`:**  
   Take last **K** valid races (chronological), regress `pace_z ~ t` via `np.polyfit(t, z, 1)`; use the slope.

5. **`team_dev_trend`:**  
   Same as above but on the team mean pace; then **broadcast** the team’s slope to its current drivers.

6. **`stability_delta_vs_tm`:**  
   For each past race: `delta = pace_z_driver − pace_z_team_mean`.  
   Over last **W**, compute robust σ: `IQR(delta)/1.35` (fallback to std if IQR=0).

7. **Robustness & fallbacks:**  
   If <3 valid points → return `NaN` for the affected trends. If no pit data → keep all laps. If no team mapping → `team_dev_trend = NaN`.

---

## Recommended preprocessing (model side)

- **Winsorize/clip** each feature to p1–p99 (computed on **train**), then **robust-scale** (median/MAD).  
- Do **not** `fit` scalers on inference.

---

## Sanity checks

- Increasing **K** (e.g., 6→10) should make trends smoother, smaller magnitudes.  
- `stability_delta_vs_tm` should **decrease** for drivers with consistent deltas to team.  
- No current-weekend files should ever be read; features must change if you decrement `round`.