#!/usr/bin/env python3
"""
Build pre‑race features (and optional targets) for one or many race weekends.

Key changes vs. old builder:
- Smarter race discovery: laps_* → fallback to results_*_* / qualifying_*_* → fallback to races.csv
- Pass‑through tuning: --verbose and --ctx-json '{...}' merged into ctx
- Optional strict mode for empty modules: --strict-empty
- Merge strategy toggle: --merge-how {outer,inner}
- Optional dump of intermediate module outputs per race to out/tmp/YYYY_R/

Outputs per race (to --out-dir, default ./out):
  features_YYYY_R.parquet  (+ .csv if --also-csv)
  targets_YYYY_R.parquet   (+ .csv) when --with-targets
Additionally: concatenated all_features.parquet and (if requested) all_targets.parquet.

Assumes package layout like:
  from features import FEATURIZERS, TARGETIZERS
and each module exposes featurize(ctx) -> DataFrame keyed by 'Driver'.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# import registry and orchestrator
from src.features import FEATURIZERS, TARGETIZERS  # type: ignore


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def parse_year_round(name: str) -> Tuple[int, int]:
    """Parse year/round from filenames like 'laps_2024_5.csv', 'results_2025_16.csv'.
    Strategy: take the last '(YYYY)_(R)' pair in the string.
    """
    m = list(re.finditer(r"(19|20)\d{2}_(\d{1,2})", name))
    if not m:
        raise ValueError(f"Cannot parse year/round from: {name}")
    year = int(m[-1].group(0).split("_")[0])
    rnd = int(m[-1].group(2))
    return year, rnd


def discover_races(raw_dir: Path, races_filter: Optional[List[str]] = None) -> List[Tuple[int, int]]:
    """Return sorted unique list of (year, round).
    Sources in order: laps_*.csv → results_*_*.csv / qualifying_*_*.csv → races.csv.
    If races_filter is given (as ['YYYY_R', ...]) keep only those tags.
    """
    allow = set(races_filter or [])
    files: List[Path] = []
    for pat in ("laps_*.csv", "results_*_*.csv", "qualifying_*_*.csv"):
        files.extend(sorted(raw_dir.glob(pat)))
    races: List[Tuple[int, int]] = []
    for p in files:
        try:
            y, r = parse_year_round(p.name)
        except Exception:
            continue
        tag = f"{y}_{r}"
        if allow and tag not in allow:
            continue
        races.append((y, r))
    if not races:
        races_csv = raw_dir / "races.csv"
        if races_csv.exists():
            df = pd.read_csv(races_csv)
            if {"year", "round"}.issubset(df.columns):
                for _, row in df.iterrows():
                    y, r = int(row["year"]), int(row["round"])  # type: ignore[index]
                    tag = f"{y}_{r}"
                    if allow and tag not in allow:
                        continue
                    races.append((y, r))
    return sorted(set(races), key=lambda x: (x[0], x[1]))


def safe_to_parquet_or_csv(df: pd.DataFrame, path: Path, *, also_csv: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Parquet
    df.to_parquet(path, index=False)
    if also_csv:
        df.to_csv(path.with_suffix(".csv"), index=False)


def run_modules(
    ctx: Dict,
    module_specs: List[Tuple[str, callable]],
    *,
    select: Optional[Iterable[str]] = None,
    dump_dir: Optional[Path] = None,
    strict_empty: bool = False,
    merge_how: str = "outer",
    verbose: bool = False,
) -> pd.DataFrame:
    """Execute selected modules and merge on 'Driver'. Optionally dump intermediates.
    Returns merged DataFrame (may be empty).
    """
    to_run = [(n, f) for (n, f) in module_specs if (select is None or n in select)]
    frames: List[pd.DataFrame] = []
    for name, fn in to_run:
        try:
            part = fn(ctx)
        except Exception as e:
            if strict_empty:
                raise
            log(f"  - {name}: EXCEPTION {e}", quiet=not verbose)
            continue
        if part is None or part.empty:
            msg = f"  - {name}: empty"
            if strict_empty:
                raise RuntimeError(msg)
            log(msg, quiet=not verbose)
            continue
        # Ensure 'Driver' present
        if "Driver" not in part.columns:
            msg = f"  - {name}: missing 'Driver' column"
            if strict_empty:
                raise RuntimeError(msg)
            log(msg, quiet=not verbose)
            continue
        # Dump
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            safe_to_parquet_or_csv(part, dump_dir / f"{name}.parquet", also_csv=True)
        frames.append(part)
        log(f"  - {name}: ok ({part.shape[0]}x{part.shape[1]})", quiet=not verbose)

    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Driver", how=merge_how)
    return out


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build pre-race features and optional targets")
    ap.add_argument("--raw-dir", required=True, help="Directory with raw CSVs (races.csv, results.csv, etc.)")
    ap.add_argument("--out-dir", default="out", help="Output directory (default: ./out)")
    ap.add_argument("--modules", default=None, help="Comma-separated subset of module names to run (from FEATURIZERS)")
    ap.add_argument("--with-targets", action="store_true", help="Also build and merge targets from TARGETIZERS")
    ap.add_argument("--races", default=None, help="Comma-separated list of tags YYYY_R to build only these races")
    ap.add_argument("--skip-existing", action="store_true", help="Skip race if features_YYYY_R already exists")
    ap.add_argument("--also-csv", action="store_true", help="Save CSV alongside Parquet")
    ap.add_argument("--dump-intermediate", action="store_true", help="Dump each module output to out/tmp/YYYY_R/")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs and ctx['verbose']=True")
    ap.add_argument("--ctx-json", type=str, default=None, help="JSON with extra ctx (weights, knobs, etc.)")
    ap.add_argument("--merge-how", type=str, default="outer", choices=["outer", "inner"], help="Merge strategy")
    ap.add_argument("--strict-empty", action="store_true", help="Treat empty module as an error")

    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    races_filter = None
    if args.races:
        races_filter = [s.strip() for s in args.races.split(",") if s.strip()]

    races = discover_races(raw, races_filter)
    if not races:
        log("No races found.")
        return

    # module selection
    selected_modules = None
    if args.modules:
        selected_modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    all_feat: List[pd.DataFrame] = []
    all_tgt: List[pd.DataFrame] = []

    for (year, rnd) in races:
        tag = f"{year}_{rnd}"
        feat_path = out_dir / f"features_{tag}.parquet"
        if args.skip_existing and feat_path.exists():
            log(f"[skip] {tag} features already exist at {feat_path}")
            continue

        log(f"\n=== Build {tag} ===")
        # dump directory per race if requested
        dump_dir = (out_dir / "tmp" / tag) if args.dump_intermediate else None

        # base ctx
        ctx: Dict = {"raw_dir": raw, "year": year, "round": rnd}
        if args.verbose:
            ctx["verbose"] = True
        if args.ctx_json:
            try:
                ctx.update(json.loads(args.ctx_json))
            except Exception as e:
                log(f"  ! bad --ctx-json: {e}")

        # run feature modules
        feat_df = run_modules(
            ctx,
            FEATURIZERS,
            select=selected_modules,
            dump_dir=dump_dir,
            strict_empty=args.strict_empty,
            merge_how=args.merge_how,
            verbose=args.verbose,
        )
        if feat_df is None or feat_df.empty:
            log("  ! no features produced (empty result)")
            continue

        # optional targets
        if args.with_targets:
            tgt_df = run_modules(
                ctx,
                TARGETIZERS,
                select=None,
                dump_dir=dump_dir,
                strict_empty=False,
                merge_how="outer",
                verbose=args.verbose,
            )
            if tgt_df is not None and not tgt_df.empty:
                # merge into features and also save standalone targets
                feat_df = feat_df.merge(tgt_df, on="Driver", how="left")
                safe_to_parquet_or_csv(tgt_df, out_dir / f"targets_{tag}.parquet", also_csv=args.also_csv)
                all_tgt.append(tgt_df.assign(year=year, round=rnd))

        # Save per race
        safe_to_parquet_or_csv(feat_df, feat_path, also_csv=args.also_csv)
        all_feat.append(feat_df.assign(year=year, round=rnd))

    # Save concatenated
    if all_feat:
        full_feat = pd.concat(all_feat, ignore_index=True, sort=False)
        safe_to_parquet_or_csv(full_feat, out_dir / "all_features.parquet", also_csv=args.also_csv)
        log(f"\nSaved {len(all_feat)} race files and all_features.parquet")
    if all_tgt and args.with_targets:
        full_tgt = pd.concat(all_tgt, ignore_index=True, sort=False)
        safe_to_parquet_or_csv(full_tgt, out_dir / "all_targets.parquet", also_csv=args.also_csv)
        log(f"Saved {len(all_tgt)} target files and all_targets.parquet")


if __name__ == "__main__":
    main()
