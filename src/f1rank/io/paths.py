from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    raw: Path
    interim: Path
    features: Path
    artifacts: Path

@dataclass
class Config:
    seed: int
    paths: Paths
    split: dict
    target: dict
    features: dict
    eval: dict

def load_config(path: str | Path) -> Config:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text())
    base = Path.cwd()  # assume run from repo root
    return Config(
        seed=cfg.get("seed", 42),
        paths=Paths(
            raw=base / cfg["paths"]["raw"],
            interim=base / cfg["paths"]["interim"],
            features=base / cfg["paths"]["features"],
            artifacts=base / cfg["paths"]["artifacts"],
        ),
        split=cfg.get("split", {}),
        target=cfg.get("target", {}),
        features=cfg.get("features", {}),
        eval=cfg.get("eval", {}),
    )
