from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

OUTCOME_LABELS: tuple[str, ...] = ("finish", "dnf", "dsq")
OUTCOME_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
FINISH_ID = OUTCOME_TO_ID["finish"]
DNF_ID = OUTCOME_TO_ID["dnf"]
DSQ_ID = OUTCOME_TO_ID["dsq"]

_FINISH_RE = re.compile(r"^(?:\+?\d+\s+LAPS?)$")
_DSQ_TOKENS: tuple[str, ...] = ("DSQ", "DISQUAL")
_DNF_TOKENS: tuple[str, ...] = (
    "DNF",
    "DNS",
    "DNQ",
    "NC",
    "NOT CLASSIFIED",
    "NOTCLASSIFIED",
    "RETIRED",
    "WITHDREW",
    "ACCIDENT",
    "COLLISION",
    "SPUN OFF",
    "SPIN",
    "ENGINE",
    "GEARBOX",
    "TRANSMISSION",
    "BRAKES",
    "BRAKE",
    "SUSPENSION",
    "HYDRAULIC",
    "ELECTRICAL",
    "POWER UNIT",
    "COOLING",
    "PUNCTURE",
    "DAMAGE",
    "OVERHEAT",
)


def is_finish_status(status: object, finish_position: object = None) -> bool:
    label = normalize_outcome_status(status, finish_position=finish_position)
    return label == "finish"


def normalize_outcome_status(status: object, finish_position: object = None) -> str:
    raw = "" if status is None else str(status).strip().upper()
    pos = pd.to_numeric(pd.Series([finish_position]), errors="coerce").iloc[0]

    if raw and raw != "NAN":
        if any(tok in raw for tok in _DSQ_TOKENS):
            return "dsq"
        if "FINISHED" in raw or "PLUS" in raw or _FINISH_RE.match(raw):
            return "finish"
        if any(tok in raw for tok in _DNF_TOKENS):
            return "dnf"

    if pd.notna(pos) and float(pos) > 0:
        return "finish"
    return "dnf"


def outcome_label_series(status: Sequence[object] | pd.Series, finish_position: Sequence[object] | pd.Series) -> pd.Series:
    pos_series = pd.Series(finish_position)
    if isinstance(status, pd.Series):
        status_series = status.reindex(pos_series.index)
    elif np.isscalar(status) or status is None:
        status_series = pd.Series([status] * len(pos_series), index=pos_series.index)
    else:
        status_series = pd.Series(status, index=pos_series.index if len(pos_series) else None)
    labels = [
        normalize_outcome_status(st, finish_position=pos)
        for st, pos in zip(status_series.tolist(), pos_series.tolist())
    ]
    return pd.Series(labels, index=status_series.index, dtype="object")


def outcome_id_series(status: Sequence[object] | pd.Series, finish_position: Sequence[object] | pd.Series) -> pd.Series:
    labels = outcome_label_series(status, finish_position)
    return labels.map(OUTCOME_TO_ID).astype("int64")


def outcome_priority(values: Sequence[object] | pd.Series) -> pd.Series:
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        mapping = {FINISH_ID: 2.0, DNF_ID: 1.0, DSQ_ID: 0.0}
        return series.map(mapping).astype(float)
    mapping = {"finish": 2.0, "dnf": 1.0, "dsq": 0.0}
    return series.astype(str).str.lower().map(mapping).astype(float)


__all__ = [
    "OUTCOME_LABELS",
    "OUTCOME_TO_ID",
    "FINISH_ID",
    "DNF_ID",
    "DSQ_ID",
    "is_finish_status",
    "normalize_outcome_status",
    "outcome_label_series",
    "outcome_id_series",
    "outcome_priority",
]
