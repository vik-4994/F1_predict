#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _years_or_default(years: Iterable[int] | None, last_n_seasons: int) -> List[int]:
    if years:
        return sorted(set(int(y) for y in years))
    cur = date.today().year
    return list(range(cur - int(last_n_seasons) + 1, cur + 1))


def _race_tags_from_logs(logs) -> List[str]:
    tags = {
        f"{int(row['year'])}_{int(row['round'])}"
        for row in logs
        if str(row.get("status", "")).startswith("ok")
    }
    return sorted(tags, key=lambda tag: tuple(int(x) for x in tag.split("_")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh recent FastF1 raw data and rebuild feature tables.")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw_csv"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/fastf1_cache"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--years", nargs="+", type=int, default=None, help="Years to refresh. Default: last two seasons.")
    ap.add_argument("--last-n-seasons", type=int, default=2)
    ap.add_argument("--sessions", nargs="+", default=["R", "Q", "S", "SQ", "FP2", "FP3"])
    ap.add_argument("--telemetry-stride", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--driver-limit", type=int, default=None)
    ap.add_argument("--latest-only", action="store_true", help="Refresh only the latest completed round per year.")
    ap.add_argument(
        "--lookback-rounds",
        type=int,
        default=2,
        help="Refresh the last N completed rounds per year before rebuilding features.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip already exported raw files.")
    ap.add_argument("--with-targets", action="store_true", help="Rebuild targets alongside features.")
    ap.add_argument("--ctx-json", type=str, default=None, help="Extra build context passed through to build_features.py.")
    ap.add_argument("--no-build", action="store_true", help="Only refresh raw FastF1 data.")
    ap.add_argument("--audit", action="store_true", help="Run feature audit after rebuild.")
    ap.add_argument("--progress", action="store_true", help="Show progress bars during feature build.")
    args = ap.parse_args()

    years = _years_or_default(args.years, args.last_n_seasons)

    export_mod = _load_script_module(
        "refresh_export_fastf1_module",
        ROOT / "scripts" / "export_last_two_years.py",
    )
    logs = export_mod.run_export(
        out_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        years=years,
        sessions=args.sessions,
        telemetry_stride=args.telemetry_stride,
        max_workers=args.max_workers,
        driver_limit=args.driver_limit,
        skip_existing=args.skip_existing,
        completed_only=True,
        latest_only=args.latest_only,
        lookback_rounds=args.lookback_rounds,
    )

    tags = _race_tags_from_logs(logs)
    if args.no_build:
        print(f"Refreshed raw data for {len(tags)} race tags: {', '.join(tags) if tags else 'none'}")
        return

    if not tags:
        print("No refreshed race tags were produced; skipping build.")
        return

    build_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_features.py"),
        "--raw-dir",
        str(args.raw_dir),
        "--out-dir",
        str(args.processed_dir),
        "--races",
        ",".join(tags),
        "--recombine-existing",
    ]
    if args.with_targets:
        build_cmd.append("--with-targets")
    if args.progress:
        build_cmd.append("--progress")
    if args.ctx_json:
        build_cmd.extend(["--ctx-json", args.ctx_json])
    _run(build_cmd)

    if args.audit:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "audit_features.py"),
                "--features",
                str(args.processed_dir / "all_features.parquet"),
            ]
        )


if __name__ == "__main__":
    main()
