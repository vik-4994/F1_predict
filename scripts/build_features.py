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
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

                                  
from src.features import ALL_FEATURIZERS, FEATURIZERS, TARGETIZERS
from src.frame_utils import sanitize_frame_columns

import warnings
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)


                                                              
         
                                                              

def log(msg: str, *, quiet: bool = False, use_tqdm: bool = False) -> None:
    if not quiet:
        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg, flush=True)


def _concat_saved_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    usable = [df for df in frames if df is not None and not df.empty]
    if not usable:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
            category=FutureWarning,
        )
        return pd.concat(usable, ignore_index=True, sort=False)


def _make_bar(
    *,
    total: int,
    desc: str,
    unit: str,
    leave: bool,
    position: int,
) -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,
        smoothing=0.1,
    )


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
             
    df.to_parquet(path, index=False)
    if also_csv:
        df.to_csv(path.with_suffix(".csv"), index=False)


def collect_saved_race_frames(
    out_dir: Path,
    prefix: str,
    *,
    skip_tags: Optional[set[str]] = None,
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(out_dir.glob(f"{prefix}_*.parquet")):
        try:
            year, rnd = parse_year_round(path.name)
        except Exception:
            continue
        tag = f"{year}_{rnd}"
        if skip_tags and tag in skip_tags:
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["year"] = int(year)
        df["round"] = int(rnd)
        frames.append(df)
    return frames


def _coalesce_overlap(left: pd.Series, right: pd.Series, name: str) -> pd.Series:
    mask = left.notna() & right.notna()
    if bool(mask.any()):
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            compatible = np.allclose(
                pd.to_numeric(left[mask], errors="coerce").to_numpy(dtype=float, copy=False),
                pd.to_numeric(right[mask], errors="coerce").to_numpy(dtype=float, copy=False),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            )
        else:
            compatible = bool((left[mask].astype(str) == right[mask].astype(str)).all())
        if not compatible:
            raise RuntimeError(f"Conflicting feature column '{name}' across modules")

    out = left.copy()
    fill_mask = out.isna() & right.notna()
    if bool(fill_mask.any()):
        out.loc[fill_mask] = right.loc[fill_mask]
    return out


def _validate_frame(name: str, part: pd.DataFrame) -> pd.DataFrame:
    part = sanitize_frame_columns(part)
    if "Driver" not in part.columns:
        raise RuntimeError(f"{name}: missing 'Driver' column")
    if part["Driver"].isna().any():
        raise RuntimeError(f"{name}: null Driver values are not allowed")
    if part["Driver"].astype(str).duplicated().any():
        raise RuntimeError(f"{name}: duplicate Driver rows are not allowed")
    return part


def _merge_parts(left: pd.DataFrame, right: pd.DataFrame, how: str) -> pd.DataFrame:
    overlap = [c for c in right.columns if c != "Driver" and c in left.columns]
    merged = left.merge(
        right,
        on="Driver",
        how=how,
        validate="one_to_one",
        suffixes=("", "__dup"),
    )
    for col in overlap:
        dup_col = f"{col}__dup"
        merged[col] = _coalesce_overlap(merged[col], merged[dup_col], col)
        merged = merged.drop(columns=[dup_col])
    return sanitize_frame_columns(merged)


def run_modules(
    ctx: Dict,
    module_specs: List[Tuple[str, callable]],
    *,
    select: Optional[Iterable[str]] = None,
    dump_dir: Optional[Path] = None,
    strict_empty: bool = False,
    merge_how: str = "outer",
    verbose: bool = False,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> pd.DataFrame:
    """Execute selected modules and merge on 'Driver'. Optionally dump intermediates.
    Returns merged DataFrame (may be empty).
    """
    to_run = [(n, f) for (n, f) in module_specs if (select is None or n in select)]
    frames: List[pd.DataFrame] = []
    ok_count = 0
    empty_count = 0
    err_count = 0

    bar = None
    if progress and not verbose:
        bar = _make_bar(
            total=len(to_run),
            desc=progress_desc or "Modules",
            unit="module",
            leave=False,
            position=1,
        )

    for name, fn in to_run:
        if bar is not None:
            bar.set_postfix_str(f"module={name} ok={ok_count} empty={empty_count} err={err_count}")
        try:
            part = fn(ctx)
        except Exception as e:
            err_count += 1
            if bar is not None:
                bar.update(1)
            if strict_empty:
                if bar is not None:
                    bar.close()
                raise
            log(
                f"{progress_desc or 'modules'} | {name}: EXCEPTION {e}",
                quiet=not verbose and not progress,
                use_tqdm=(progress and not verbose),
            )
            continue
        if part is None or part.empty:
            msg = f"  - {name}: empty"
            empty_count += 1
            if bar is not None:
                bar.update(1)
            if strict_empty:
                if bar is not None:
                    bar.close()
                raise RuntimeError(msg)
            log(msg, quiet=not verbose)
            continue
        try:
            part = _validate_frame(name, part)
        except Exception as e:
            err_count += 1
            if bar is not None:
                bar.update(1)
            if strict_empty:
                if bar is not None:
                    bar.close()
                raise
            log(
                f"{progress_desc or 'modules'} | {name}: EXCEPTION {e}",
                quiet=not verbose and not progress,
                use_tqdm=(progress and not verbose),
            )
            continue
              
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            safe_to_parquet_or_csv(part, dump_dir / f"{name}.parquet", also_csv=True)
        frames.append(part)
        ok_count += 1
        if bar is not None:
            bar.update(1)
        log(f"  - {name}: ok ({part.shape[0]}x{part.shape[1]})", quiet=not verbose)

    if bar is not None:
        bar.set_postfix_str(f"ok={ok_count} empty={empty_count} err={err_count} merge")
        bar.close()

    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = _merge_parts(out, df, merge_how)
    return out


                                                              
     
                                                              

def main() -> None:
    ap = argparse.ArgumentParser(description="Build pre-race features and optional targets")
    ap.add_argument("--raw-dir", required=True, help="Directory with raw CSVs (races.csv, results.csv, etc.)")
    ap.add_argument("--out-dir", default="out", help="Output directory (default: ./out)")
    ap.add_argument("--modules", default=None, help="Comma-separated subset of module names to run (from FEATURIZERS)")
    ap.add_argument("--with-targets", action="store_true", help="Also build and save standalone targets from TARGETIZERS")
    ap.add_argument("--races", default=None, help="Comma-separated list of tags YYYY_R to build only these races")
    ap.add_argument("--skip-existing", action="store_true", help="Skip race if features_YYYY_R already exists")
    ap.add_argument("--also-csv", action="store_true", help="Save CSV alongside Parquet")
    ap.add_argument("--dump-intermediate", action="store_true", help="Dump each module output to out/tmp/YYYY_R/")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs and ctx['verbose']=True")
    ap.add_argument("--ctx-json", type=str, default=None, help="JSON with extra ctx (weights, knobs, etc.)")
    ap.add_argument("--merge-how", type=str, default="outer", choices=["outer", "inner"], help="Merge strategy")
    ap.add_argument("--strict-empty", action="store_true", help="Treat empty module as an error")
    ap.add_argument("--progress", action="store_true", help="Show progress bars for races/modules")
    ap.add_argument(
        "--recombine-existing",
        action="store_true",
        help="When writing all_features/all_targets, include already saved per-race files from --out-dir",
    )

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

                      
    selected_modules = None
    if args.modules:
        selected_modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    all_feat: List[pd.DataFrame] = []
    all_tgt: List[pd.DataFrame] = []
    built_tags: set[str] = set()
    built_count = 0
    skipped_count = 0
    empty_count = 0

    race_bar = None
    if args.progress and not args.verbose:
        race_bar = _make_bar(
            total=len(races),
            desc="Races",
            unit="race",
            leave=True,
            position=0,
        )

    for (year, rnd) in races:
        tag = f"{year}_{rnd}"
        if race_bar is not None:
            race_bar.set_description(f"Race {tag}")
            race_bar.set_postfix_str(f"built={built_count} skipped={skipped_count} empty={empty_count}")
        feat_path = out_dir / f"features_{tag}.parquet"
        if args.skip_existing and feat_path.exists():
            skipped_count += 1
            log(
                f"[skip] {tag} features already exist at {feat_path}",
                quiet=(args.progress and not args.verbose),
                use_tqdm=(args.progress and not args.verbose),
            )
            if race_bar is not None:
                race_bar.update(1)
            continue

        log(f"\n=== Build {tag} ===", quiet=(args.progress and not args.verbose))
                                              
        dump_dir = (out_dir / "tmp" / tag) if args.dump_intermediate else None

                  
        ctx: Dict = {"raw_dir": raw, "year": year, "round": rnd}
        if args.verbose:
            ctx["verbose"] = True
        if args.ctx_json:
            try:
                ctx.update(json.loads(args.ctx_json))
            except Exception as e:
                log(
                    f"  ! bad --ctx-json: {e}",
                    use_tqdm=(args.progress and not args.verbose),
                )

                             
        feat_df = run_modules(
            ctx,
            ALL_FEATURIZERS if selected_modules is not None else FEATURIZERS,
            select=selected_modules,
            dump_dir=dump_dir,
            strict_empty=args.strict_empty,
            merge_how=args.merge_how,
            verbose=args.verbose,
            progress=(args.progress and not args.verbose),
            progress_desc=f"{tag} features",
        )
        if feat_df is None or feat_df.empty:
            empty_count += 1
            log(
                f"{tag} | no features produced (empty result)",
                use_tqdm=(args.progress and not args.verbose),
            )
            if race_bar is not None:
                race_bar.update(1)
            continue

                          
        if args.with_targets:
            if race_bar is not None:
                race_bar.set_postfix_str(f"built={built_count} skipped={skipped_count} empty={empty_count} stage=targets")
            tgt_df = run_modules(
                ctx,
                TARGETIZERS,
                select=None,
                dump_dir=dump_dir,
                strict_empty=False,
                merge_how="outer",
                verbose=args.verbose,
                progress=(args.progress and not args.verbose),
                progress_desc=f"{tag} targets",
            )
            if tgt_df is not None and not tgt_df.empty:
                tgt_df = _validate_frame("targets", tgt_df)
                safe_to_parquet_or_csv(tgt_df, out_dir / f"targets_{tag}.parquet", also_csv=args.also_csv)
                all_tgt.append(tgt_df.assign(year=year, round=rnd))

                       
        safe_to_parquet_or_csv(feat_df, feat_path, also_csv=args.also_csv)
        all_feat.append(feat_df.assign(year=year, round=rnd))
        built_tags.add(tag)
        built_count += 1
        if race_bar is not None:
            race_bar.set_postfix_str(
                f"built={built_count} skipped={skipped_count} empty={empty_count} rows={len(feat_df)}"
            )
            race_bar.update(1)

    if race_bar is not None:
        race_bar.set_description("Races done")
        race_bar.set_postfix_str(f"built={built_count} skipped={skipped_count} empty={empty_count}")
        race_bar.close()

    feat_frames = list(all_feat)
    if args.recombine_existing:
        feat_frames = collect_saved_race_frames(out_dir, "features", skip_tags=built_tags) + feat_frames
    if feat_frames:
        full_feat = _concat_saved_frames(feat_frames)
        safe_to_parquet_or_csv(full_feat, out_dir / "all_features.parquet", also_csv=args.also_csv)
        log(f"\nSaved {len(feat_frames)} race feature frames into all_features.parquet")

    tgt_frames = list(all_tgt)
    if args.recombine_existing and args.with_targets:
        tgt_frames = collect_saved_race_frames(out_dir, "targets", skip_tags=built_tags) + tgt_frames
    if tgt_frames and args.with_targets:
        full_tgt = _concat_saved_frames(tgt_frames)
        safe_to_parquet_or_csv(full_tgt, out_dir / "all_targets.parquet", also_csv=args.also_csv)
        log(f"Saved {len(tgt_frames)} target frames into all_targets.parquet")


if __name__ == "__main__":
    main()
