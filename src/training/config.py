# FILE: src/training/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import argparse
import json
import os
import time


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


@dataclass
class TrainConfig:
    # ---- paths ----
    features_path: Path = Path("data/processed/all_features.parquet")
    targets_path: Path = Path("data/processed/all_targets.parquet")
    out_dir: Path = Path("models")
    run_name: str = "ranker_v1"          # итоговые артефакты попадут в out_dir / run_name

    # ---- split ----
    val_last: int = 6                    # сколько последних гонок отдать на валидацию

    # ---- optimization ----
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: List[int] = None             # наполним из строки "256,128" ниже
    dropout: float = 0.10
    seed: int = 42

    # ---- device ----
    device: str = "auto"                 # "auto" | "cpu" | "cuda"

    # ---- misc ----
    log_every: int = 1                   # как часто печатать метрики по эпохам
    dnf_position: int = 21               # куда отправлять DNF в порядке (чем больше, тем хуже)

    def artifacts_dir(self) -> Path:
        """models/<run_name> под out_dir."""
        return self.out_dir / self.run_name

    # -------- utilities --------
    def to_dict(self) -> dict:
        d = asdict(self)
        # путь к конечной папке (удобно при логировании)
        d["artifacts_dir"] = str(self.artifacts_dir())
        # сериализация путей в строки
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
    # paths
    ap.add_argument("--features", default="data/processed/all_features.parquet",
                    help="Путь к all_features.parquet")
    ap.add_argument("--targets", default="data/processed/all_targets.parquet",
                    help="Путь к all_targets.parquet")
    ap.add_argument("--out-dir", default="models",
                    help="Базовая папка для артефактов")
    ap.add_argument("--run-name", default="ranker_v1",
                    help="Имя подпапки с артефактами в out-dir (models/<run-name>)")
    # split
    ap.add_argument("--val-last", type=int, default=6,
                    help="Сколько последних гонок использовать для валидации")
    # optimization
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=_parse_hidden, default=_parse_hidden("256,128"),
                    help='Скрытые слои MLP через запятую, напр. "512,256,128"')
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    # device & misc
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--log-every", type=int, default=1, help="Частота логирования (в эпохах)")
    ap.add_argument("--dnf-position", type=int, default=21,
                    help="Эффективная позиция для DNF в тренировочном порядке")
    return ap


def from_args(args: Optional[argparse.Namespace] = None) -> TrainConfig:
    """
    Создаёт TrainConfig из argparse.Namespace (или парсит sys.argv, если None).
    """
    ap = build_argparser()
    ns = ap.parse_args([] if args is not None else None) if isinstance(args, argparse.Namespace) else ap.parse_args()

    # Пришёл ли готовый Namespace?
    if isinstance(args, argparse.Namespace):
        ns = args  # уважаем внешний парсинг

    cfg = TrainConfig(
        features_path=Path(ns.features),
        targets_path=Path(ns.targets),
        out_dir=Path(ns.out_dir),
        run_name=str(ns.run_name),
        val_last=int(ns.val_last),
        epochs=int(ns.epochs),
        lr=float(ns.lr),
        weight_decay=float(ns.weight_decay),
        hidden=list(ns.hidden) if ns.hidden is not None else [256, 128],
        dropout=float(ns.dropout),
        seed=int(ns.seed),
        device=str(ns.device),
        log_every=int(ns.log_every),
        dnf_position=int(ns.dnf_position),
    )

    # Если run-name = "auto", создадим метку по времени
    if cfg.run_name in ("auto", "AUTO"):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cfg.run_name = f"ranker_{stamp}"

    return cfg


__all__ = ["TrainConfig", "build_argparser", "from_args"]
