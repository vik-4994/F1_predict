from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scenario_builder import (
    build_scenario_features,
    resolve_official_track_name,
    resolve_scenario_mode,
)
from src.scenario_support import is_artifact_compatible, resolve_artifacts_dir

POINTS_BY_RANK: Dict[int, int] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Simulate season standings from per-race predictions")
    ap.add_argument("--artifacts", type=str, required=True)
    ap.add_argument("--future-artifacts", type=str, default=None, help="Optional future-compatible artifacts dir")
    ap.add_argument("--raw-dir", type=str, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--rounds", type=str, default=None, help='Optional CSV/range like "1,2,5-8"')
    ap.add_argument(
        "--scenario-mode",
        type=str,
        default="auto",
        choices=["auto", "observed", "future"],
        help="auto=use observed weekend data when available, otherwise priors-only future mode",
    )
    ap.add_argument("--tau", type=float, default=1.2, help="softmax temperature")
    ap.add_argument("--out-dir", type=str, default=None, help="Optional directory for CSV exports")
    ap.add_argument("--race-topk", type=int, default=3, help="How many drivers to show per race summary")
    return ap.parse_args(argv)


def _parse_rounds_arg(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    out: List[int] = []
    for chunk in value.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(part))
    return sorted(set(out))


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
    rounds = rounds[(rounds > 0) & rounds.notna()]
    return sorted({int(r) for r in rounds.tolist()})


def _available_rounds_from_raw(raw_dir: Path, year: int) -> List[int]:
    rounds: set[int] = set()
    for pattern in (
        f"entrylist_{year}_*_Q.csv",
        f"entrylist_{year}_*.csv",
        f"results_{year}_*_Q.csv",
        f"results_{year}_*.csv",
        f"meta_{year}_*.csv",
    ):
        for path in raw_dir.glob(pattern):
            match = re.search(rf"_{year}_(\d{{1,2}})(?:_[A-Za-z0-9]+)?\.csv$", path.name)
            if match:
                rounds.add(int(match.group(1)))
    return sorted(rounds)


def resolve_rounds(raw_dir: Path, year: int, rounds_arg: Optional[str], allow_future: bool = False) -> List[int]:
    requested = _parse_rounds_arg(rounds_arg)
    schedule_rounds = _load_schedule_rounds(raw_dir, year)
    available_rounds = _available_rounds_from_raw(raw_dir, year)

    if requested is not None:
        base = requested
    elif allow_future and schedule_rounds:
        base = schedule_rounds
    elif schedule_rounds and available_rounds:
        base = [rnd for rnd in schedule_rounds if rnd in set(available_rounds)]
    else:
        base = schedule_rounds or available_rounds
    return [rnd for rnd in base if int(rnd) > 0]


def _find_grid_col(cols: Iterable[str]) -> Optional[str]:
    lowered = {str(col).strip().lower(): col for col in cols}
    for alias in GRID_ALIASES:
        if alias in lowered:
            return lowered[alias]
    for alias in GRID_ALIASES:
        for norm, col in lowered.items():
            if alias in norm:
                return col
    return None


def _team_map_for_round(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    pairs = [(int(year), int(rnd))]
    pairs.extend((int(year), r) for r in range(int(rnd) - 1, 0, -1))
    for y in range(int(year) - 1, int(year) - 4, -1):
        if y > 0:
            pairs.extend((y, r) for r in range(24, 0, -1))

    seen: set[tuple[int, int]] = set()
    for y, r in pairs:
        if (y, r) in seen:
            continue
        seen.add((y, r))
        for name in (
            f"results_{y}_{r}_Q.csv",
            f"entrylist_{y}_{r}_Q.csv",
            f"results_{y}_{r}.csv",
            f"entrylist_{y}_{r}.csv",
        ):
            path = raw_dir / name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            dcol = next((c for c in ("Abbreviation", "Driver", "code", "driverRef", "BroadcastName") if c in df.columns), None)
            tcol = next((c for c in ("TeamName", "Team", "Constructor", "ConstructorName") if c in df.columns), None)
            if dcol and tcol:
                out = pd.DataFrame({"Driver": df[dcol].astype(str), "Team": df[tcol].astype(str)})
                out = out.dropna(subset=["Driver"]).drop_duplicates("Driver")
                if not out.empty:
                    return out
    return pd.DataFrame(columns=["Driver", "Team"])


def assign_race_points(rank_df: pd.DataFrame) -> pd.DataFrame:
    out = rank_df.copy()
    out["points"] = out["rank"].map(POINTS_BY_RANK).fillna(0).astype(int)
    out["win"] = (out["rank"] == 1).astype(int)
    out["podium"] = out["rank"].isin([1, 2, 3]).astype(int)
    return out


def _race_summary_line(rank_df: pd.DataFrame, track_name: str, topk: int) -> str:
    show = rank_df.nsmallest(int(topk), "rank")[["Driver", "points"]].copy()
    bits = [f"{row.Driver} {int(row.points)}" for row in show.itertuples(index=False)]
    return f"R{int(rank_df['round'].iloc[0]):02d} {track_name}: " + " | ".join(bits)


def _driver_standings(preds: pd.DataFrame) -> pd.DataFrame:
    agg = (
        preds.groupby("Driver", as_index=False)
        .agg(points=("points", "sum"), wins=("win", "sum"), podiums=("podium", "sum"), races=("round", "nunique"))
        .sort_values(["points", "wins", "podiums", "Driver"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    agg["rank"] = np.arange(1, len(agg) + 1, dtype=np.int32)
    return agg[["rank", "Driver", "points", "wins", "podiums", "races"]]


def _team_standings(preds: pd.DataFrame) -> pd.DataFrame:
    if "Team" not in preds.columns or preds["Team"].dropna().empty:
        return pd.DataFrame()
    agg = (
        preds.dropna(subset=["Team"])
        .groupby("Team", as_index=False)
        .agg(points=("points", "sum"), wins=("win", "sum"), podiums=("podium", "sum"), races=("round", "nunique"))
        .sort_values(["points", "wins", "podiums", "Team"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    agg["rank"] = np.arange(1, len(agg) + 1, dtype=np.int32)
    return agg[["rank", "Team", "points", "wins", "podiums", "races"]]


def _print_table(title: str, df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    print(f"\n{title}")
    show = df if max_rows is None else df.head(int(max_rows))
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(show.to_string(index=False))


def main(argv: Optional[Sequence[str]] = None) -> None:
    from src.training import InferenceRunner

    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    rounds = resolve_rounds(
        raw_dir,
        int(args.year),
        args.rounds,
        allow_future=(str(args.scenario_mode).lower() in {"auto", "future"}),
    )
    if not rounds:
        raise SystemExit(f"No available rounds found for {args.year}")

    base_artifacts_dir = Path(args.artifacts)
    runners: Dict[Path, InferenceRunner] = {}
    all_preds: List[pd.DataFrame] = []
    skipped: List[str] = []

    for rnd in rounds:
        official_track = resolve_official_track_name(raw_dir, int(args.year), int(rnd))
        scenario_mode = resolve_scenario_mode(raw_dir, int(args.year), int(rnd), args.scenario_mode)
        try:
            feat_df, track_name, _ = build_scenario_features(
                raw_dir,
                int(args.year),
                int(rnd),
                track=official_track,
                drivers=None,
                mode="auto",
                scenario_mode=scenario_mode,
                allow_fallback_actual=True,
                verbose=False,
            )
        except Exception as exc:
            skipped.append(f"R{int(rnd):02d}: {exc}")
            continue

        try:
            effective_artifacts_dir = resolve_artifacts_dir(
                base_artifacts_dir,
                scenario_mode,
                future_artifacts_dir=args.future_artifacts,
            )
        except Exception as exc:
            skipped.append(f"R{int(rnd):02d}: {exc}")
            continue
        runner = runners.get(effective_artifacts_dir)
        if runner is None:
            runner = InferenceRunner.from_dir(effective_artifacts_dir)
            runners[effective_artifacts_dir] = runner
        if not is_artifact_compatible(runner.artifacts.meta, scenario_mode):
            skipped.append(
                f"R{int(rnd):02d}: artifacts at {effective_artifacts_dir} are not compatible with mode '{scenario_mode}'"
            )
            continue

        rank_df = runner.rank(
            feat_df,
            temperature=float(args.tau),
            by=("year", "round"),
            include_probs=True,
            ascending=False,
        )
        rank_df = assign_race_points(rank_df)
        rank_df["track"] = str(track_name or official_track or f"Round {rnd}")
        rank_df["scenario_mode"] = scenario_mode

        grid_col = _find_grid_col(feat_df.columns)
        if grid_col is not None:
            grid_map = (
                feat_df[["Driver", grid_col]]
                .drop_duplicates("Driver")
                .set_index("Driver")[grid_col]
            )
            rank_df["grid"] = pd.to_numeric(rank_df["Driver"].map(grid_map), errors="coerce")

        team_map = _team_map_for_round(raw_dir, int(args.year), int(rnd))
        if not team_map.empty:
            rank_df = rank_df.merge(team_map, on="Driver", how="left")

        label = f"{str(rank_df['track'].iloc[0])} [{scenario_mode}]"
        print(_race_summary_line(rank_df, label, int(args.race_topk)))
        all_preds.append(rank_df)

    if not all_preds:
        skipped_msg = "\n".join(skipped) if skipped else "No rounds could be simulated"
        raise SystemExit(skipped_msg)

    preds = pd.concat(all_preds, ignore_index=True, sort=False)
    preds = preds.sort_values(["round", "rank"], kind="mergesort").reset_index(drop=True)
    driver_standings = _driver_standings(preds)
    team_standings = _team_standings(preds)

    _print_table("Driver Standings", driver_standings)
    if not team_standings.empty:
        _print_table("Team Standings", team_standings)

    if skipped:
        print("\nSkipped rounds")
        for item in skipped:
            print(item)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_dir / f"season_predictions_{args.year}.csv", index=False)
        driver_standings.to_csv(out_dir / f"driver_standings_{args.year}.csv", index=False)
        if not team_standings.empty:
            team_standings.to_csv(out_dir / f"team_standings_{args.year}.csv", index=False)
        print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
