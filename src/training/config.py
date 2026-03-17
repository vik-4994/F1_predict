                              
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional
import argparse
import json
import time

from src.scenario_support import FULL_FEATURE_PROFILE, FUTURE_FEATURE_PROFILE


def _parse_hidden(s: str) -> List[int]:
    """
    Parse hidden sizes like "256,128,64" -> [256, 128, 64].
    Empty string -> [].
    """
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid hidden layer size: '{tok}'")
    return out


def _parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _parse_float_mapping(s: str) -> dict[str, float]:
    text = (s or "").strip()
    if not text:
        return {}
    out: dict[str, float] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid mapping item: '{item}'")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Invalid mapping item: '{item}'")
        try:
            out[key] = float(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid float value in mapping item: '{item}'") from exc
    return out


BASELINE_HIDDEN = [128, 64]
BASELINE_DROPOUT = 0.25
BASELINE_LR = 7e-4
BASELINE_WEIGHT_DECAY = 3e-4
BASELINE_EPOCHS = 16
BASELINE_DROP_PREFIXES = ["pitcrew_", "slowstop_"]
BASELINE_DROP_CONTAINS = ["double_stack", "undercut", "overcut"]
BASELINE_DROP_COLS = ["expected_stop_count", "first_stint_len_exp"]
FUTURE_TRAIN_RECENCY_HALF_LIFE = 8.0


@dataclass
class TrainConfig:
                     
    features_path: Path = Path("data/processed/all_features.parquet")
    targets_path: Path = Path("data/processed/all_targets.parquet")
    out_dir: Path = Path("models")
    run_name: str = "ranker_v1"                                                           

                     
    val_last: int = 6                                                                 

                            
    epochs: int = BASELINE_EPOCHS
    lr: float = BASELINE_LR
    weight_decay: float = BASELINE_WEIGHT_DECAY
    hidden: List[int] = field(default_factory=lambda: list(BASELINE_HIDDEN))
    dropout: float = BASELINE_DROPOUT
    seed: int = 42

                      
    device: str = "auto"                                          

                    
    log_every: int = 1                                                         
    dnf_position: int = 21                                                                     
    dsq_position: int = 25
    status_loss_weight: float = 1.0
    drop_prefixes: List[str] = field(default_factory=lambda: list(BASELINE_DROP_PREFIXES))
    drop_contains: List[str] = field(default_factory=lambda: list(BASELINE_DROP_CONTAINS))
    drop_cols: List[str] = field(default_factory=lambda: list(BASELINE_DROP_COLS))
    keep_prefixes: List[str] = field(default_factory=list)
    feature_profile: str = FULL_FEATURE_PROFILE
    train_recency_half_life: Optional[float] = None
    use_regulation_era_weights: bool = True
    era_weights: dict[str, float] = field(default_factory=dict)

    def artifacts_dir(self) -> Path:
        """Return models/<run_name> under out_dir."""
        return self.out_dir / self.run_name

                                 
    def to_dict(self) -> dict:
        d = asdict(self)
                                                        
        d["artifacts_dir"] = str(self.artifacts_dir())
                                     
        d["features_path"] = str(self.features_path)
        d["targets_path"] = str(self.targets_path)
        d["out_dir"] = str(self.out_dir)
        return d

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Train ranker config")
           
    ap.add_argument("--features", default="data/processed/all_features.parquet",
                    help="Path to all_features.parquet")
    ap.add_argument("--targets", default="data/processed/all_targets.parquet",
                    help="Path to all_targets.parquet")
    ap.add_argument("--out-dir", default="models",
                    help="Base directory for saved artifacts")
    ap.add_argument("--run-name", default="ranker_v1",
                    help="Artifact subdirectory name inside out-dir (models/<run-name>)")
           
    ap.add_argument("--val-last", type=int, default=6,
                    help="How many latest races to keep for validation")
                  
    ap.add_argument("--epochs", type=int, default=BASELINE_EPOCHS)
    ap.add_argument("--lr", type=float, default=BASELINE_LR)
    ap.add_argument("--weight-decay", type=float, default=BASELINE_WEIGHT_DECAY)
    ap.add_argument("--hidden", type=_parse_hidden, default=list(BASELINE_HIDDEN),
                    help='MLP hidden sizes as CSV, e.g. "512,256,128"')
    ap.add_argument("--dropout", type=float, default=BASELINE_DROPOUT)
    ap.add_argument("--seed", type=int, default=42)
                   
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--log-every", type=int, default=1, help="Logging frequency in epochs")
    ap.add_argument("--dnf-position", type=int, default=21,
                    help="Effective finishing position used for DNF ordering during training")
    ap.add_argument("--dsq-position", type=int, default=25,
                    help="Effective finishing position used for DSQ ordering during training/eval")
    ap.add_argument("--status-loss-weight", type=float, default=1.0,
                    help="Weight for the finish/DNF/DSQ classification loss")
    ap.add_argument("--drop-prefixes", type=_parse_csv_list, default=list(BASELINE_DROP_PREFIXES),
                    help='Drop features by prefix, e.g. "tele_pre_,pitcrew_"')
    ap.add_argument("--drop-contains", type=_parse_csv_list, default=list(BASELINE_DROP_CONTAINS),
                    help='Drop features by substring, e.g. "double_stack,track_cluster_"')
    ap.add_argument("--drop-cols", type=_parse_csv_list, default=list(BASELINE_DROP_COLS),
                    help='Drop exact feature names, e.g. "track_cluster_id,undercut_window_width"')
    ap.add_argument("--keep-prefixes", type=_parse_csv_list, default=[],
                    help='Keep only features with these prefixes before drop filters are applied')
    ap.add_argument(
        "--feature-profile",
        choices=["full", "future"],
        default=FULL_FEATURE_PROFILE,
        help="Training feature profile: full=standard baseline, future=priors-only future mode",
    )
    ap.add_argument(
        "--train-recency-half-life",
        type=float,
        default=None,
        help=(
            "Optional train-time race recency half-life. "
            f"Default: disabled for {FULL_FEATURE_PROFILE}, {FUTURE_TRAIN_RECENCY_HALF_LIFE:g} for {FUTURE_FEATURE_PROFILE}."
        ),
    )
    ap.add_argument(
        "--use-regulation-era-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply regulation-era race weights on top of recency weighting.",
    )
    ap.add_argument(
        "--era-weights",
        type=_parse_float_mapping,
        default={},
        help=(
            "Optional era weight overrides, e.g. "
            '"next_gen_2026=1.0,ground_effect=0.75,wide_aero_hybrid=0.5,aero_kers=0.18"'
        ),
    )
    return ap


def from_args(args: Optional[argparse.Namespace] = None) -> TrainConfig:
    """Build TrainConfig from argparse.Namespace or parse sys.argv if args is None."""
    ap = build_argparser()
    ns = ap.parse_args([] if args is not None else None) if isinstance(args, argparse.Namespace) else ap.parse_args()

                                  
    if isinstance(args, argparse.Namespace):
        ns = args                           

    cfg = TrainConfig(
        features_path=Path(ns.features),
        targets_path=Path(ns.targets),
        out_dir=Path(ns.out_dir),
        run_name=str(ns.run_name),
        val_last=int(ns.val_last),
        epochs=int(ns.epochs),
        lr=float(ns.lr),
        weight_decay=float(ns.weight_decay),
        hidden=list(ns.hidden) if ns.hidden is not None else list(BASELINE_HIDDEN),
        dropout=float(ns.dropout),
        seed=int(ns.seed),
        device=str(ns.device),
        log_every=int(ns.log_every),
        dnf_position=int(ns.dnf_position),
        dsq_position=int(ns.dsq_position),
        status_loss_weight=float(ns.status_loss_weight),
        drop_prefixes=list(ns.drop_prefixes or []),
        drop_contains=list(ns.drop_contains or []),
        drop_cols=list(ns.drop_cols or []),
        keep_prefixes=list(ns.keep_prefixes or []),
        feature_profile=str(ns.feature_profile),
        train_recency_half_life=(
            None if ns.train_recency_half_life is None else float(ns.train_recency_half_life)
        ),
        use_regulation_era_weights=bool(ns.use_regulation_era_weights),
        era_weights=dict(ns.era_weights or {}),
    )

                                                       
    if cfg.run_name in ("auto", "AUTO"):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cfg.run_name = f"ranker_{stamp}"

    return cfg


__all__ = ["TrainConfig", "build_argparser", "from_args"]
