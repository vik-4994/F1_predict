#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_audit import column_health_report, group_health_report


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit feature-table health by column and group")
    ap.add_argument("--features", default="data/processed/all_features.parquet", help="Feature table (.parquet or .csv)")
    ap.add_argument("--out-dir", default="reports", help="Directory for audit CSVs")
    ap.add_argument("--top", type=int, default=25, help="How many worst columns/groups to print")
    args = ap.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_table(features_path)
    col_report = column_health_report(df)
    grp_report = group_health_report(col_report)

    col_path = out_dir / "feature_health_columns.csv"
    grp_path = out_dir / "feature_health_groups.csv"
    col_report.to_csv(col_path, index=False)
    grp_report.to_csv(grp_path, index=False)

    top_n = max(1, int(args.top))
    print(f"rows={len(df)} cols={len(df.columns)}")
    print(f"saved columns report to {col_path}")
    print(f"saved groups report to {grp_path}")

    dead_groups = grp_report.loc[grp_report["dead_group"]]
    if not dead_groups.empty:
        print("\nDead groups:")
        print(dead_groups[["group", "cols", "all_nan_cols"]].to_string(index=False))

    worst_groups = grp_report.head(top_n)
    if not worst_groups.empty:
        print("\nWorst groups:")
        print(
            worst_groups[
                ["group", "cols", "all_nan_cols", "mean_missing_rate", "mean_race_constant_share"]
            ].to_string(index=False)
        )

    worst_cols = col_report.head(top_n)
    if not worst_cols.empty:
        print("\nWorst columns:")
        print(
            worst_cols[
                ["feature", "group", "all_nan", "missing_rate", "race_constant_share", "unique_non_na"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
