from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

FUTURE_MODE = "future"
FULL_FEATURE_PROFILE = "full"
FUTURE_FEATURE_PROFILE = "future"
SUPPORTED_FEATURE_PROFILES = {FULL_FEATURE_PROFILE, FUTURE_FEATURE_PROFILE}

# Prefixes produced by the current priors-only modules used in future mode.
_FUTURE_FEATURE_PREFIXES: tuple[str, ...] = (
    "track_",
    "track_is_",
    "driver_trackc_pre_",
    "weather_pre_",
    "chaos_pre_",
    "hist_pre_",
    "tele_pre_",
    "quali_pre_",
    "compound_mix_priors_",
    "tyre_delta_priors_",
    "expected_deg_",
    "reliab_",
    "lap1_",
    "net_pass_",
    "driver_team_pre_",
)

_FUTURE_FEATURE_EXACT: set[str] = {
    "driver_trend",
    "team_dev_trend",
    "stability_delta_vs_tm",
}

_DEFAULT_FUTURE_MODEL_CANDIDATES: tuple[str, ...] = (
    "baseline_future_v1",
    "ranker_future_v1",
)


def normalize_feature_profile(value: Optional[str]) -> str:
    profile = str(value or FULL_FEATURE_PROFILE).strip().lower()
    if profile not in SUPPORTED_FEATURE_PROFILES:
        raise ValueError(f"Unsupported feature profile: {value}")
    return profile


def is_future_feature_col(name: str) -> bool:
    col = str(name).strip()
    return col in _FUTURE_FEATURE_EXACT or any(col.startswith(prefix) for prefix in _FUTURE_FEATURE_PREFIXES)


def select_feature_profile_cols(feature_cols: Sequence[str], profile: str) -> List[str]:
    normalized = normalize_feature_profile(profile)
    cols = [str(col) for col in feature_cols]
    if normalized != FUTURE_FEATURE_PROFILE:
        return cols
    kept = [col for col in cols if is_future_feature_col(col)]
    if not kept:
        raise ValueError("Future feature profile removed all feature columns")
    return kept


def artifact_feature_profile(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return FULL_FEATURE_PROFILE
    profile = meta.get("feature_profile")
    if isinstance(profile, str) and profile.strip().lower() in SUPPORTED_FEATURE_PROFILES:
        return profile.strip().lower()
    scenario_mode = str(meta.get("scenario_mode", "")).strip().lower()
    return FUTURE_FEATURE_PROFILE if scenario_mode == FUTURE_MODE else FULL_FEATURE_PROFILE


def is_artifact_compatible(meta: Optional[Dict[str, Any]], scenario_mode: str) -> bool:
    mode = str(scenario_mode or "").strip().lower()
    if mode != FUTURE_MODE:
        return True
    return artifact_feature_profile(meta) == FUTURE_FEATURE_PROFILE


def load_artifact_meta(artifacts_dir: Path | str) -> Dict[str, Any]:
    meta_path = Path(artifacts_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_artifacts_dir(
    artifacts_dir: Path | str,
    scenario_mode: str,
    *,
    future_artifacts_dir: Optional[Path | str] = None,
) -> Path:
    base_dir = Path(artifacts_dir)
    mode = str(scenario_mode or "").strip().lower()
    if mode != FUTURE_MODE:
        return base_dir

    if future_artifacts_dir:
        return Path(future_artifacts_dir)

    if is_artifact_compatible(load_artifact_meta(base_dir), mode):
        return base_dir

    parent = base_dir.parent
    candidates: List[Path] = [
        parent / f"{base_dir.name}_future",
        parent / f"{base_dir.name}_future_v1",
    ]
    candidates.extend(parent / name for name in _DEFAULT_FUTURE_MODEL_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        if is_artifact_compatible(load_artifact_meta(candidate), mode):
            return candidate

    raise RuntimeError(
        "Future scenario requires a future-compatible artifacts directory. "
        "Pass --future-artifacts or train a model with --feature-profile future."
    )


__all__ = [
    "FUTURE_FEATURE_PROFILE",
    "FUTURE_MODE",
    "FULL_FEATURE_PROFILE",
    "SUPPORTED_FEATURE_PROFILES",
    "artifact_feature_profile",
    "is_artifact_compatible",
    "is_future_feature_col",
    "load_artifact_meta",
    "normalize_feature_profile",
    "resolve_artifacts_dir",
    "select_feature_profile_cols",
]
