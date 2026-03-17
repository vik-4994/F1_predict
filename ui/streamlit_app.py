from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frame_utils import sanitize_frame_columns
from src.scenario_builder import (
    build_scenario_features,
    resolve_official_track_name,
    resolve_scenario_mode,
)
from src.scenario_support import is_artifact_compatible, resolve_artifacts_dir
from src.training import InferenceRunner
from src.ui_jobs import (
    DEFAULT_FUTURE_RUN_NAME,
    ROOT as PROJECT_ROOT,
    active_job,
    future_artifacts_dir,
    read_status,
    season_output_dir,
)


RAW_DIR = ROOT / "data" / "raw_csv"
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = "baseline_v4"
DEFAULT_SEASON_SCENARIO = "future"
BUY_ME_A_COFFEE_URL = os.getenv("BUY_ME_A_COFFEE_URL", "").strip()
ALPHA_NUM = "abcdefghijklmnopqrstuvwxyz0123456789"
GRID_ALIASES = [
    "grid",
    "grid_pos",
    "grid_position",
    "start_grid",
    "start_grid_pos",
    "starting_grid",
    "start_position",
    "quali_grid",
    "quali_grid_pos",
]
WEATHER_MAP: Dict[str, List[str]] = {
    "AirTemp": ["weather_air_temp", "air_temp", "air_temp_c", "air_temperature_c", "met_air_c", "weather::air_temp"],
    "TrackTemp": ["weather_track_temp", "track_temp", "track_temp_c", "track_temperature_c", "weather::track_temp"],
    "Humidity": ["weather_humidity", "rel_humidity", "relative_humidity", "humidity"],
    "WindSpeed": ["weather_wind_speed", "wind_speed", "wind_speed_mps", "wind_speed_kph", "wind_mps", "wind_kph"],
    "WindDirection": ["weather_wind_dir", "wind_direction", "wind_direction_deg", "wind_dir"],
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-main: #f4efe6;
            --panel: rgba(255, 252, 247, 0.86);
            --panel-strong: rgba(255, 250, 243, 0.96);
            --ink: #191919;
            --muted: #5c5b57;
            --accent: #d83d17;
            --accent-2: #0e5a73;
            --line: rgba(25, 25, 25, 0.08);
            --shadow: 0 18px 50px rgba(106, 72, 43, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(216, 61, 23, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 90, 115, 0.12), transparent 32%),
                linear-gradient(180deg, #f7f1e8 0%, #f3ebdf 45%, #efe5d6 100%);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        h1, h2, h3, p, label, div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        code, pre {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 247, 236, 0.95), rgba(247, 238, 226, 0.95));
            border-right: 1px solid var(--line);
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,252,247,0.95), rgba(255,244,231,0.92));
            border: 1px solid rgba(216, 61, 23, 0.12);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.6rem 1.7rem 1.4rem 1.7rem;
            margin-bottom: 1rem;
            animation: fadeUp 420ms ease-out;
        }

        .hero-kicker {
            display: inline-block;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--accent-2);
            margin-bottom: 0.7rem;
            font-weight: 700;
        }

        .hero-title {
            font-size: clamp(2rem, 3.2vw, 3.6rem);
            line-height: 0.96;
            margin: 0 0 0.5rem 0;
            font-weight: 700;
            max-width: 760px;
        }

        .hero-copy {
            max-width: 760px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        .hero-actions {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .cta, .cta-secondary {
            display: inline-block;
            padding: 0.78rem 1rem;
            border-radius: 999px;
            text-decoration: none;
            font-weight: 700;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .cta {
            background: linear-gradient(135deg, #d83d17, #f06a22);
            color: white !important;
            box-shadow: 0 14px 28px rgba(216, 61, 23, 0.18);
        }

        .cta-secondary {
            color: var(--accent-2) !important;
            border: 1px solid rgba(14, 90, 115, 0.18);
            background: rgba(255, 255, 255, 0.52);
        }

        .cta:hover, .cta-secondary:hover {
            transform: translateY(-1px);
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            animation: fadeUp 520ms ease-out;
        }

        .stDataFrame, div[data-testid="stMetric"] {
            background: var(--panel-strong);
            border-radius: 18px;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _norm(value: str) -> str:
    out: List[str] = []
    prev_us = False
    for ch in value.strip().lower():
        if ch in ALPHA_NUM:
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    if out and out[-1] == "_":
        out.pop()
    return "".join(out)


def _find_col(all_cols: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    cols = list(all_cols)
    cols_norm = {c: _norm(c) for c in cols}
    norm_to_col = {v: k for k, v in cols_norm.items()}
    for alias in aliases:
        key = _norm(alias)
        if key in norm_to_col:
            return norm_to_col[key]
    for alias in aliases:
        key = _norm(alias)
        for col, norm_col in cols_norm.items():
            if key in norm_col:
                return col
    return None


def _onehot_track_columns(cols: Iterable[str]) -> List[str]:
    return [c for c in cols if str(c).startswith(("trk::", "track::", "track_is_"))]


def _resolve_track_onehot_col(track_onehots: Sequence[str], track_name: str) -> Optional[str]:
    base_to_col: Dict[str, str] = {}
    for col in track_onehots:
        base = str(col)
        for prefix in ("trk::", "track::", "track_is_"):
            if base.startswith(prefix):
                base = base[len(prefix) :]
        base_to_col[_norm(base)] = str(col)

    target = _norm(track_name)
    if target in base_to_col:
        return base_to_col[target]

    best, score = None, 0.0
    for base in base_to_col:
        ratio = difflib.SequenceMatcher(a=base, b=target).ratio()
        if ratio > score:
            best, score = base, ratio
    return base_to_col[best] if best and score >= 0.80 else None


def _apply_track_onehot(df: pd.DataFrame, track_name: str) -> None:
    onehots = _onehot_track_columns(df.columns)
    if not onehots:
        return
    df.loc[:, onehots] = np.float32(0.0)
    matched_col = _resolve_track_onehot_col(onehots, track_name)
    if matched_col:
        df.loc[:, matched_col] = np.float32(1.0)


def _apply_weather(df: pd.DataFrame, weather_json: Dict[str, float]) -> None:
    if not weather_json:
        return
    cols = list(df.columns)
    for key, aliases in WEATHER_MAP.items():
        if key not in weather_json:
            continue
        col = _find_col(cols, aliases)
        if col is not None:
            df.loc[:, col] = np.float32(weather_json[key])


def _apply_grid(df: pd.DataFrame, grid_map: Dict[str, float]) -> None:
    if not grid_map:
        return
    col = _find_col(df.columns, GRID_ALIASES)
    if col is None:
        return
    for drv, pos in grid_map.items():
        mask = df["Driver"].astype(str) == str(drv)
        if bool(mask.any()):
            df.loc[mask, col] = np.float32(pos)


def _build_grid_pos_map(df_features: pd.DataFrame, user_grid_map: Dict[str, float]) -> Dict[str, float]:
    if user_grid_map:
        return {str(key): float(value) for key, value in user_grid_map.items()}
    col = _find_col(df_features.columns, GRID_ALIASES)
    if col is None:
        return {}
    mapping_df = df_features[["Driver", col]].dropna().copy()
    mapping_df[col] = pd.to_numeric(mapping_df[col], errors="coerce")
    mapping_df = mapping_df.dropna()
    return {str(row["Driver"]): float(row[col]) for _, row in mapping_df.iterrows()}


def _softmax_stable(values: np.ndarray, tau: float) -> np.ndarray:
    temp = float(tau) if tau > 0 else 1.0
    shifted = values - np.max(values)
    exp_values = np.exp(shifted / temp)
    denom = exp_values.sum()
    return exp_values / denom if denom > 0 else np.full_like(values, 1.0 / max(1, len(values)))


def _apply_grid_weight(
    rank_df: pd.DataFrame,
    df_features: pd.DataFrame,
    user_grid_map: Dict[str, float],
    grid_weight: float,
    tau: float,
    beta: float = 0.25,
) -> pd.DataFrame:
    if grid_weight <= 0:
        return rank_df

    grid_map = _build_grid_pos_map(df_features, user_grid_map)
    if not grid_map:
        return rank_df

    df = rank_df.copy()
    df["__grid_pos__"] = df["Driver"].map(grid_map).fillna(10.0)
    penalty = float(grid_weight) * float(beta) * (df["__grid_pos__"] - 1.0)
    base_rank_col = "score_rank" if "score_rank" in df.columns else "score"
    df[base_rank_col] = df[base_rank_col] - penalty

    if "score_status" in df.columns and "score_rank" in df.columns:
        rank_vals = df["score_rank"].to_numpy(dtype=np.float64)
        centered = rank_vals - rank_vals.mean()
        sigma = rank_vals.std()
        if sigma > 1e-6:
            centered = np.clip(centered / sigma, -3.0, 3.0)
        df["score"] = df["score_status"].to_numpy(dtype=np.float64) * 10.0 + centered
        if "p_finish" in df.columns:
            probs = _softmax_stable(
                df["score_rank"].to_numpy(dtype=np.float64)
                + np.log(np.clip(df["p_finish"].to_numpy(dtype=np.float64), 1e-6, 1.0)),
                tau=float(tau),
            )
        else:
            probs = _softmax_stable(df["score"].to_numpy(dtype=np.float64), tau=float(tau))
    else:
        df["score"] = df[base_rank_col].to_numpy(dtype=np.float64)
        probs = _softmax_stable(df["score"].to_numpy(dtype=np.float64), tau=float(tau))
    df["p_win"] = probs.astype(np.float32)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=np.int32)
    return df.drop(columns=["__grid_pos__"])


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_grid(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        try:
            out[key.strip().upper()] = float(raw_value)
        except ValueError:
            continue
    return out


def _load_schedule_rounds(raw_dir: Path, year: int) -> List[int]:
    schedule_path = raw_dir / f"schedule_{year}.csv"
    if not schedule_path.exists():
        return []
    sched = pd.read_csv(schedule_path)
    if sched.empty:
        return []
    round_col = "round" if "round" in sched.columns else ("RoundNumber" if "RoundNumber" in sched.columns else None)
    if round_col is None:
        return []
    rounds = pd.to_numeric(sched[round_col], errors="coerce")
    return sorted({int(rnd) for rnd in rounds.dropna().tolist() if int(rnd) > 0})


def _available_rounds_from_raw(raw_dir: Path, year: int) -> List[int]:
    rounds: set[int] = set()
    patterns = (
        f"entrylist_{year}_*.csv",
        f"results_{year}_*.csv",
        f"meta_{year}_*.csv",
    )
    for pattern in patterns:
        for path in raw_dir.glob(pattern):
            match = re.search(rf"_{year}_(\d{{1,2}})(?:_[A-Za-z0-9]+)?\.csv$", path.name)
            if match:
                rounds.add(int(match.group(1)))
    return sorted(rounds)


def _available_years(raw_dir: Path) -> List[int]:
    years: set[int] = set()
    for path in raw_dir.glob("schedule_*.csv"):
        match = re.search(r"schedule_(\d{4})\.csv$", path.name)
        if match:
            years.add(int(match.group(1)))
    for path in raw_dir.glob("meta_*.csv"):
        match = re.search(r"meta_(\d{4})_\d{1,2}(?:_[A-Za-z0-9]+)?\.csv$", path.name)
        if match:
            years.add(int(match.group(1)))
    return sorted(years)


def _rounds_for_year(raw_dir: Path, year: int) -> List[int]:
    rounds = set(_load_schedule_rounds(raw_dir, year))
    rounds.update(_available_rounds_from_raw(raw_dir, year))
    return sorted(rounds)


def _discover_models(models_dir: Path) -> List[str]:
    if not models_dir.exists():
        return []
    out: List[str] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        files = {path.name for path in child.iterdir() if path.is_file()}
        needed = {"ranker.pt", "scaler.json", "feature_cols.txt"}
        if needed.issubset(files):
            out.append(child.name)
    return out


def _launch_background_job(job: str, year: int, model_name: str, future_run_name: str = DEFAULT_FUTURE_RUN_NAME) -> None:
    subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ui_job_runner.py"),
            "--job",
            str(job),
            "--year",
            str(int(year)),
            "--baseline-artifacts",
            str(MODELS_DIR / model_name),
            "--future-run-name",
            str(future_run_name),
        ],
        cwd=str(PROJECT_ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@st.cache_data(show_spinner=False)
def _load_season_outputs(year: int, scenario_mode: str = DEFAULT_SEASON_SCENARIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = season_output_dir(int(year), scenario_mode)
    driver_path = out_dir / f"driver_standings_{int(year)}.csv"
    team_path = out_dir / f"team_standings_{int(year)}.csv"
    season_path = out_dir / f"season_predictions_{int(year)}.csv"

    driver_df = pd.read_csv(driver_path) if driver_path.exists() else pd.DataFrame()
    team_df = pd.read_csv(team_path) if team_path.exists() else pd.DataFrame()
    season_df = pd.read_csv(season_path) if season_path.exists() else pd.DataFrame()
    return driver_df, team_df, season_df


def _render_ops_panel(model_name: str, year: int) -> None:
    status = read_status()
    active = active_job()
    is_running = bool(active)
    effective_status = active if active else status

    st.markdown("## Season Ops")
    st.markdown(
        """
        <div class="panel">
            Launch long-running local jobs from the UI: refresh FastF1 data, train the future model, and simulate the full season.
            The jobs run in the background and write logs plus status to the workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    if cols[0].button("Refresh Data", use_container_width=True, disabled=is_running):
        _launch_background_job("refresh", int(year), model_name)
        st.rerun()
    if cols[1].button("Train Future Model", use_container_width=True, disabled=is_running):
        _launch_background_job("train_future", int(year), model_name)
        st.rerun()
    if cols[2].button("Simulate Season", use_container_width=True, disabled=is_running):
        _launch_background_job("simulate_future", int(year), model_name)
        st.rerun()
    if cols[3].button("Run Full Pipeline", type="primary", use_container_width=True, disabled=is_running):
        _launch_background_job("full_future", int(year), model_name)
        st.rerun()

    refresh_col, info_col = st.columns([0.25, 0.75])
    if refresh_col.button("Refresh Status", use_container_width=True):
        st.rerun()

    if effective_status:
        state = str(effective_status.get("state", "idle")).upper()
        step = str(effective_status.get("current_step") or "-")
        info_col.caption(
            f"State: {state} | Step: {step} | Started: {effective_status.get('started_at', '-')}"
        )
        log_path = effective_status.get("log_path")
        message = str(effective_status.get("message", ""))
        if message:
            if state == "FAILED":
                st.error(message)
            elif state == "SUCCEEDED":
                st.success(message)
            else:
                st.info(message)
        if log_path and Path(str(log_path)).exists():
            with st.expander("Job log", expanded=False):
                log_text = Path(str(log_path)).read_text(encoding="utf-8", errors="replace")
                st.code(log_text[-12000:] if len(log_text) > 12000 else log_text, language="text")
    else:
        info_col.caption("No background jobs have been started yet.")


def _render_season_dashboard(year: int, scenario_mode: str = DEFAULT_SEASON_SCENARIO) -> None:
    driver_df, team_df, season_df = _load_season_outputs(int(year), scenario_mode)
    out_dir = season_output_dir(int(year), scenario_mode)

    st.markdown("## Season Projection")
    if driver_df.empty:
        st.warning(
            f"No season projection found for {year}. Run 'Simulate Season' or 'Run Full Pipeline' to generate "
            f"{out_dir.name}."
        )
        return

    metrics = st.columns(4)
    leader = driver_df.iloc[0]
    metrics[0].metric("Projected champion", str(leader.get("Driver", "-")))
    metrics[1].metric("Projected points", int(leader.get("points", 0)))
    metrics[2].metric("Drivers ranked", len(driver_df))
    metrics[3].metric("Scenario", scenario_mode.upper())

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown("### Driver Standings")
        st.dataframe(driver_df, use_container_width=True, hide_index=True)
    with right:
        if not team_df.empty:
            st.markdown("### Team Standings")
            st.dataframe(team_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("### Team Standings")
            st.caption("Team standings are not available for this projection.")

    if not season_df.empty:
        race_points = (
            season_df.groupby(["round", "Driver"], as_index=False)["points"].sum()
            .sort_values(["round", "points", "Driver"], ascending=[True, False, True])
        )
        st.markdown("### Race-by-Race Points")
        st.dataframe(race_points, use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner=False)
def _load_runner(artifacts_dir: str) -> InferenceRunner:
    return InferenceRunner.from_dir(artifacts_dir)


@st.cache_data(show_spinner=False)
def _prepare_base_features(
    raw_dir: str,
    year: int,
    rnd: int,
    scenario_mode: str,
    track: str,
    drivers_csv: str,
) -> Tuple[pd.DataFrame, str, List[str], str]:
    requested_track = track.strip() or None
    drivers = _parse_csv_list(drivers_csv)
    resolved_mode = resolve_scenario_mode(raw_dir, int(year), int(rnd), scenario_mode)
    df, resolved_track, roster = build_scenario_features(
        raw_dir=raw_dir,
        sim_year=int(year),
        sim_round=int(rnd),
        track=requested_track,
        drivers=drivers or None,
        mode="auto",
        scenario_mode=resolved_mode,
        allow_fallback_actual=True,
        verbose=False,
    )
    final_track = requested_track or resolved_track or resolve_official_track_name(raw_dir, int(year), int(rnd)) or ""
    return df, final_track, roster, resolved_mode


def _format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_hero() -> None:
    donate = BUY_ME_A_COFFEE_URL or "https://www.buymeacoffee.com/"
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">Formula 1 ML Ranking</div>
            <h1 class="hero-title">Race-weekend predictions, packaged for hosting.</h1>
            <p class="hero-copy">
                This UI runs the repo's existing scenario builder and ranking model on top of raw FastF1-derived data.
                It is structured as a simple hosted product page: pick a race, generate a grid forecast, and surface a donation CTA.
            </p>
            <div class="hero-actions">
                <a class="cta" href="{donate}" target="_blank">Buy me a coffee</a>
                <a class="cta-secondary" href="https://github.com/vik-4994/F1_predict" target="_blank">View source</a>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(models: List[str], years: List[int]) -> Dict[str, object]:
    st.sidebar.markdown("## Race Setup")
    default_model_idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    selected_model = st.sidebar.selectbox("Model artifacts", models, index=default_model_idx if models else None)

    selected_year = st.sidebar.selectbox("Season", years, index=len(years) - 1 if years else None)
    rounds = _rounds_for_year(RAW_DIR, int(selected_year)) if selected_year else []
    default_round_idx = 0 if not rounds else len(rounds) - 1
    selected_round = st.sidebar.selectbox("Round", rounds, index=default_round_idx if rounds else None)
    scenario_mode = st.sidebar.selectbox("Scenario mode", ["auto", "future", "observed"], index=0)

    official_track = ""
    if selected_year and selected_round:
        official_track = resolve_official_track_name(RAW_DIR, int(selected_year), int(selected_round)) or ""
    track_name = st.sidebar.text_input("Track override", value=official_track)
    drivers_csv = st.sidebar.text_input("Drivers (optional CSV)", value="", placeholder="VER,NOR,LEC")

    with st.sidebar.expander("Advanced controls"):
        tau = st.slider("Softmax temperature", min_value=0.4, max_value=2.5, value=1.2, step=0.1)
        grid_weight = st.slider("Grid bias", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        grid_csv = st.text_input("Grid override", value="", placeholder="VER=1,NOR=2,PIA=3")
        air_temp = st.number_input("AirTemp", value=27.0, step=0.5)
        track_temp = st.number_input("TrackTemp", value=36.0, step=0.5)
        humidity = st.number_input("Humidity", value=52.0, step=1.0)
        wind_speed = st.number_input("WindSpeed", value=4.0, step=0.5)
        wind_direction = st.number_input("WindDirection", value=190.0, step=1.0)
        topk = st.slider("Show top drivers", min_value=3, max_value=20, value=10, step=1)

    submit = st.sidebar.button("Run prediction", type="primary", use_container_width=True)
    return {
        "selected_model": selected_model,
        "selected_year": selected_year,
        "selected_round": selected_round,
        "scenario_mode": scenario_mode,
        "track_name": track_name,
        "drivers_csv": drivers_csv,
        "tau": tau,
        "grid_weight": grid_weight,
        "grid_csv": grid_csv,
        "topk": topk,
        "weather": {
            "AirTemp": float(air_temp),
            "TrackTemp": float(track_temp),
            "Humidity": float(humidity),
            "WindSpeed": float(wind_speed),
            "WindDirection": float(wind_direction),
        },
        "submit": submit,
    }


def _run_prediction(
    model_name: str,
    year: int,
    rnd: int,
    scenario_mode: str,
    track_name: str,
    drivers_csv: str,
    tau: float,
    grid_weight: float,
    grid_csv: str,
    weather_payload: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, List[str], Dict[str, object]]:
    model_dir = MODELS_DIR / model_name
    base_df, resolved_track, roster, resolved_mode = _prepare_base_features(
        raw_dir=str(RAW_DIR),
        year=int(year),
        rnd=int(rnd),
        scenario_mode=scenario_mode,
        track=track_name,
        drivers_csv=drivers_csv,
    )
    effective_model_dir = resolve_artifacts_dir(model_dir, resolved_mode)
    runner = _load_runner(str(effective_model_dir))
    if not is_artifact_compatible(runner.artifacts.meta, resolved_mode):
        raise RuntimeError(
            f"Artifacts at {effective_model_dir} are not compatible with scenario mode '{resolved_mode}'."
        )
    features_df = base_df.copy()
    if resolved_track:
        _apply_track_onehot(features_df, resolved_track)
    _apply_weather(features_df, weather_payload)
    grid_map = _parse_grid(grid_csv)
    _apply_grid(features_df, grid_map)
    features_df = sanitize_frame_columns(features_df.copy())

    rank_df = runner.rank(
        features_df,
        temperature=float(tau),
        by=("year", "round"),
        include_probs=True,
        ascending=False,
    )
    rank_df = _apply_grid_weight(
        rank_df=rank_df,
        df_features=features_df,
        user_grid_map=grid_map,
        grid_weight=float(grid_weight),
        tau=float(tau),
    )
    meta = dict(runner.artifacts.meta)
    meta["resolved_mode"] = resolved_mode
    meta["resolved_track"] = resolved_track
    meta["effective_model_dir"] = str(effective_model_dir)
    meta["effective_model_name"] = effective_model_dir.name
    return features_df, rank_df, resolved_track, roster, meta


def _prediction_summary(rank_df: pd.DataFrame, meta: Dict[str, object], topk: int) -> None:
    leader = rank_df.iloc[0]
    podium = rank_df.head(3)["Driver"].tolist()
    metrics = st.columns(4)
    metrics[0].metric("Projected winner", str(leader["Driver"]))
    metrics[1].metric("Win probability", _format_probability(float(leader.get("p_win", 0.0))))
    metrics[2].metric("Drivers scored", f"{len(rank_df)}")
    metrics[3].metric("Scenario mode", str(meta.get("resolved_mode", "unknown")).upper())

    st.markdown(
        f"""
        <div class="panel">
            <strong>Podium forecast:</strong> {' | '.join(podium)}
            <br/>
            <strong>Validation snapshot:</strong>
            Spearman {float(meta.get('val_mean', {}).get('spearman', 0.0)):.3f},
            NDCG@5 {float(meta.get('val_mean', {}).get('ndcg5', 0.0)):.3f},
            Top-1 {float(meta.get('val_mean', {}).get('top1', 0.0)):.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_df = rank_df.copy()
    for prob_col in ("p_win", "p_finish", "p_dnf", "p_dsq"):
        if prob_col in display_df.columns:
            display_df[f"{prob_col}_%"] = (display_df[prob_col].astype(float) * 100.0).round(2)
    view_cols = [
        col
        for col in ["rank", "Driver", "predicted_outcome", "score", "p_win_%", "p_finish_%", "p_dnf_%", "p_dsq_%"]
        if col in display_df.columns
    ]
    st.dataframe(display_df[view_cols].head(int(topk)), use_container_width=True, hide_index=True)


def _render_charts(rank_df: pd.DataFrame, topk: int) -> None:
    chart_df = rank_df.head(int(topk)).copy()
    if "p_win" not in chart_df.columns:
        return
    chart_df = chart_df.set_index("Driver")[["p_win"]]
    st.bar_chart(chart_df, height=320)


def _render_footer(features_df: pd.DataFrame, rank_df: pd.DataFrame, track_name: str, roster: Sequence[str]) -> None:
    csv_payload = rank_df.to_csv(index=False).encode("utf-8")
    left, right = st.columns([1.3, 1.0])
    with left:
        st.markdown(
            f"""
            <div class="panel">
                <strong>Scenario details</strong><br/>
                Track: {track_name or 'unknown'}<br/>
                Drivers in roster: {len(roster)}<br/>
                Feature columns used: {len(features_df.columns)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.download_button(
            "Download prediction CSV",
            data=csv_payload,
            file_name=f"prediction_{track_name or 'race'}.csv".replace(" ", "_").lower(),
            mime="text/csv",
            use_container_width=True,
        )
        if BUY_ME_A_COFFEE_URL:
            st.link_button("Support this project", BUY_ME_A_COFFEE_URL, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="F1 Predict UI", page_icon="🏁", layout="wide")
    _inject_css()

    models = _discover_models(MODELS_DIR)
    years = _available_years(RAW_DIR)
    if not models:
        st.error("No model artifacts found in ./models.")
        st.stop()
    if not years:
        st.error("No schedule or metadata files found in ./data/raw_csv.")
        st.stop()

    _render_hero()
    state = _render_sidebar(models, years)
    selected_year = int(state["selected_year"])
    selected_model = str(state["selected_model"])

    ops_col, prediction_col = st.columns([0.95, 1.35], gap="large")
    with ops_col:
        _render_ops_panel(selected_model, selected_year)
        _render_season_dashboard(selected_year, DEFAULT_SEASON_SCENARIO)

    with prediction_col:
        if not state["submit"]:
            st.info("Pick a race configuration in the sidebar, then run a prediction.")
            return

        try:
            with st.spinner("Building scenario features and scoring drivers..."):
                features_df, rank_df, resolved_track, roster, meta = _run_prediction(
                    model_name=selected_model,
                    year=selected_year,
                    rnd=int(state["selected_round"]),
                    scenario_mode=str(state["scenario_mode"]),
                    track_name=str(state["track_name"]),
                    drivers_csv=str(state["drivers_csv"]),
                    tau=float(state["tau"]),
                    grid_weight=float(state["grid_weight"]),
                    grid_csv=str(state["grid_csv"]),
                    weather_payload=dict(state["weather"]),
                )
        except Exception as exc:
            st.exception(exc)
            return

        st.markdown("## Prediction")
        _prediction_summary(rank_df, meta, topk=int(state["topk"]))
        st.markdown("## Win Probability")
        _render_charts(rank_df, topk=int(state["topk"]))
        st.markdown("## Model Card")
        st.json(
            {
                "model": selected_model,
                "effective_model": meta.get("effective_model_name"),
                "track": resolved_track,
                "year": selected_year,
                "round": int(state["selected_round"]),
                "mode": meta.get("resolved_mode"),
                "features": int(meta.get("num_features", 0)),
                "best_epoch": int(meta.get("best_epoch", 0)),
                "validation": meta.get("val_mean", {}),
            },
            expanded=True,
        )
        st.markdown("## Export")
        _render_footer(features_df, rank_df, resolved_track, roster)


if __name__ == "__main__":
    main()
