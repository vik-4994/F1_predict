# F1 LLM — Track Profile Features (v2)

This page documents **every feature emitted by `track_profile.featurize(ctx)`** and provides a lean rewrite of the module with only tiny English comments.

---

## Scope

The module outputs **track-level** features duplicated per driver (one row per driver). All numeric outputs are `float32` unless noted.

---

## Base (raw) features

### `track_slug`

* **Meaning:** canonical snake-case identifier of the circuit (e.g., `singapore_grand_prix`).
* **Type/Range:** string.
* **Source:** ctx/meta/CSV detection.
* **Window:** static.
* **Effect (typical):** For tracing/debug only; not used by the model directly.

### `track_straight_pct`

* **Meaning:** share of lap spent on straights.
* **Type/Range:** [0..1].
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ benefits power/low-drag packages.

### `track_fast_corner_ratio`

* **Meaning:** share of fast corners.
* **Type/Range:** [0..1].
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ increases aero demand; favors high-DF.

### `track_lap_km`

* **Meaning:** lap length in kilometers.
* **Type/Range:** >0 (km).
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ tends to raise pit-loss and fuel effect (via derived fields).

### `track_braking_ev_per_lap`

* **Meaning:** heavy braking events per lap.
* **Type/Range:** integer-like ≥0.
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ improves overtaking potential (see `track_overtake_index`).

### `track_drs_zones`

* **Meaning:** number of DRS zones.
* **Type/Range:** integer-like ≥0.
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ improves overtaking potential.

### `track_tdeg_index`

* **Meaning:** tyre degradation severity.
* **Type/Range:** [0..1].
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ amplifies undercut power and wear-driven strategies.

### `track_aero_df_index`

* **Meaning:** aerodynamic downforce demand.
* **Type/Range:** [0..1].
* **Source:** static profile.
* **Window:** static.
* **Effect (typical):** ↑ favors high-downforce packages.

### `track_cluster`

* **Meaning:** categorical cluster of aero demand: `low_df` / `balanced` / `high_df`.
* **Type/Range:** string.
* **Source:** static mapping (CSV override allowed).
* **Window:** static.
* **Effect (typical):** coarse prior; use one-hots below.

### `track_cluster_id`

* **Meaning:** ordinal ID of the cluster (0: low_df, 1: balanced, 2: high_df).
* **Type/Range:** {0,1,2}.
* **Source:** derived from `track_cluster`.
* **Window:** static.
* **Effect (typical):** avoid for linear models; prefer one-hot features.

### `track_cluster_low_df`, `track_cluster_balanced`, `track_cluster_high_df`

* **Meaning:** one-hot encoding of `track_cluster`.
* **Type/Range:** {0,1}.
* **Source:** derived from `track_cluster`.
* **Window:** static.
* **Effect (typical):** safe categorical signal.

---

## Derived indices (normalized)

### `track_straight_n`

* **Meaning:** normalized straight share.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_straight_pct` with fixed bounds.
* **Effect (typical):** ↑ power/low-drag friendly.

### `track_fast_corner_n`

* **Meaning:** normalized fast-corner share.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_fast_corner_ratio`.
* **Effect (typical):** ↑ aero-demand.

### `track_braking_n`

* **Meaning:** normalized braking events per lap.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_braking_ev_per_lap`.
* **Effect (typical):** ↑ easier overtakes / higher energy recovery.

### `track_drs_n`

* **Meaning:** normalized DRS zones.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_drs_zones`.
* **Effect (typical):** ↑ easier overtakes.

### `track_lap_n`

* **Meaning:** normalized lap length.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_lap_km`.
* **Effect (typical):** ↑→ larger pit loss/fuel effect.

### `track_tdeg_n`

* **Meaning:** normalized tyre degradation.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_tdeg_index`.
* **Effect (typical):** ↑ strengthens tyre-limited dynamics.

### `track_aero_df_n`

* **Meaning:** normalized aero DF demand.
* **Type/Range:** [0..1].
* **Source:** min–max of `track_aero_df_index`.
* **Effect (typical):** ↑ favors high-DF packages.

---

## Composite/strategic

### `track_power_vs_df_index`

* **Meaning:** signed balance between power sensitivity and downforce demand (≈ `straight_n − 0.7*aero_n`).
* **Type/Range:** clipped to ~[-1, +1].
* **Source:** derived from normalized indices.
* **Effect (typical):** ↑ helps power tracks; ↓ helps high-DF tracks.

### `track_overtake_index`

* **Meaning:** overtaking ease proxy (mix of straights, braking, DRS, penalized by fast corners).
* **Type/Range:** [0..1].
* **Source:** weighted aggregation.
* **Effect (typical):** ↑ reduces quali/start lock-in; benefits chargers.

### `track_pit_loss_s`

* **Meaning:** expected total pit loss (in-lane + stop).
* **Type/Range:** seconds.
* **Source:** static or heuristic fallback.
* **Effect (typical):** ↑ discourages extra stops; alters under/overcut value.

### `track_fuel_effect_s_per10kg`

* **Meaning:** lap-time penalty per +10kg fuel.
* **Type/Range:** seconds per 10kg.
* **Source:** static or heuristic fallback.
* **Effect (typical):** ↑ amplifies early-stint pace gaps.

---

## Notes

* All numeric outputs are cast to `float32`.
* CSV overrides can replace both profile and cluster; missing CSV fields fall back to defaults.
* The module is **leak-safe** by construction (purely static per-circuit inputs).