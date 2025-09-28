# tools/audit_features.py — hardened v2
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# тише предупреждение из pandas groupby.apply
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning,
)

# раннер и трансформер из твоего пайплайна
try:
    from src.training.inference import InferenceRunner
except Exception as e:  # дадим явную ошибку, если путь другой
    raise

try:
    from src.training.featureset import transform_with_scaler_df  # type: ignore
    _HAS_SCALER = True
except Exception:
    _HAS_SCALER = False

BY_COLS = ("year", "round")
KEY = ("Driver",) + BY_COLS

# ---------- utils ----------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(a)
    b = pd.Series(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(a[mask].rank().corr(b[mask].rank(), method="pearson"))


def _psi(train_s: pd.Series, val_s: pd.Series, bins: int = 10) -> float:
    t = pd.to_numeric(train_s, errors="coerce")
    v = pd.to_numeric(val_s, errors="coerce")
    t = t[np.isfinite(t)]
    v = v[np.isfinite(v)]
    if t.empty or v.empty:
        return float("nan")
    eps = 1e-6
    q = np.unique(np.quantile(t, np.linspace(0, 1, bins + 1)))
    if len(q) < 2:
        return 0.0
    t_counts = np.histogram(t, bins=q)[0].astype(np.float64)
    v_counts = np.histogram(v, bins=q)[0].astype(np.float64)
    t_p = (t_counts / max(t_counts.sum(), 1.0)).clip(eps, 1)
    v_p = (v_counts / max(v_counts.sum(), 1.0)).clip(eps, 1)
    return float(np.sum((v_p - t_p) * np.log(v_p / t_p)))


def _expected_direction(col: str) -> int | None:
    s = col.lower()
    if any(k in s for k in ["_pos_p50", "_pos_iqr", "finish_p50", "finish_iqr", "pace_p50_s", "bestpos_min"]):
        return -1
    if any(k in s for k in ["_share", "_rate", "_quality", "_perf", "top", "clean_share"]):
        return +1
    return None


def _ensure_finish_eff(df: pd.DataFrame, *, target_hint: Optional[str] = None, dnf_position: int = 21) -> pd.DataFrame:
    """Гарантирует наличие 'finish_pos_eff'. Если нет — пытается вывести из target_hint или подходящих колонок.
    dnf_position ставится для DNF по столбцу Status (если есть).
    """
    if "finish_pos_eff" in df.columns:
        return df

    low = {c.lower(): c for c in df.columns}
    candidates: List[str] = []

    if target_hint and target_hint in df.columns:
        candidates.append(target_hint)
    elif target_hint and target_hint.lower() in low:
        candidates.append(low[target_hint.lower()])

    # типовые имена
    for k in [
        "finish_pos_eff", "finish_position", "finish_pos", "final_position",
        "position", "pos", "finishplace", "finish_place",
        "target_finish_position", "target_pos", "y_finish_pos", "y_pos"
    ]:
        if k in low:
            candidates.append(low[k])

    # эвристика
    for c in df.columns:
        s = c.lower()
        if ("finish" in s or s.startswith("pos") or s.endswith("_pos")) and ("pos" in s or "position" in s):
            candidates.append(c)

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise KeyError(
            "Target column not found: expected 'finish_pos_eff' or a finish-position column. "
            f"Available columns sample: {list(df.columns)[:25]}"
        )

    col = None
    for name in candidates:
        v = pd.to_numeric(df[name], errors="coerce")
        if np.isfinite(v).any():
            col = name
            break
    if col is None:
        raise KeyError("No numeric finish position column found to derive 'finish_pos_eff'")

    out = df.copy()
    fin = pd.to_numeric(out[col], errors="coerce").astype("float32")

    # DNF по Status → dnf_position
    if "Status" in out.columns:
        st = out["Status"].astype(str).str.lower()
        dnf_mask = st.str.contains(
            r"dnf|did not finish|retired|accident|collision|engine|gearbox|hydraul|electr|wheel|not classified|\bnc\b|disqual|dq|damage|drive shaft"
        )
        fin = fin.mask(dnf_mask, float(dnf_position))

    out["finish_pos_eff"] = fin.astype("float32")
    return out


def _prepare_splits(
    feats: pd.DataFrame,
    tgts: pd.DataFrame,
    val_last: int,
    *,
    target_hint: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """INNER join фич и таргета + сплит по последним гонкам (year, round)."""
    for df in (feats, tgts):
        if df.duplicated(subset=list(KEY)).any():
            df.sort_values(list(KEY), inplace=True)
            df.drop_duplicates(subset=list(KEY), keep="last", inplace=True)

    df = feats.merge(tgts, on=list(KEY), how="inner", validate="one_to_one").copy()
    df = _ensure_finish_eff(df, target_hint=target_hint)

    races = (
        df.drop_duplicates(list(BY_COLS))[list(BY_COLS)]
        .sort_values(list(BY_COLS))
        .to_numpy()
    )
    if len(races) == 0:
        raise RuntimeError("Empty join(features, targets).")

    val_cut = races[-val_last:] if val_last > 0 else races[-1:]
    val_keys = set((int(y), int(r)) for y, r in val_cut)

    val_mask = df.apply(lambda r: (int(r["year"]), int(r["round"])) in val_keys, axis=1)
    df_tr = df.loc[~val_mask].reset_index(drop=True)
    df_va = df.loc[val_mask].reset_index(drop=True)

    # гарантируем наличие finish_pos_eff после сплита
    df_tr = _ensure_finish_eff(df_tr, target_hint=target_hint)
    df_va = _ensure_finish_eff(df_va, target_hint=target_hint)

    # типы ключей
    for d in (df_tr, df_va):
        if "year" in d.columns:
            d["year"] = d["year"].astype("int32")
        if "round" in d.columns:
            d["round"] = d["round"].astype("int32")

    return df_tr, df_va


def _pick_col(df: pd.DataFrame, base: str) -> Optional[str]:
    for name in (base, f"{base}_tgt", f"{base}_x", f"{base}_y"):
        if name in df.columns:
            return name
    return None


def _score_col(df: pd.DataFrame) -> Optional[str]:
    # на разных раннерах может называться по-разному
    for name in ("score", "pred", "prediction", "yhat", "proba", "logit", "output"):
        if name in df.columns:
            return name
    return None


def _groupwise_permutation_drop(
    runner: InferenceRunner,
    df_val: pd.DataFrame,
    base_metric: float,
    feature: str,
    repeats: int = 3,
) -> float:
    """Перемешиваем feature внутри каждой гонки, считаем падение метрики (Spearman(score, -finish_pos_eff))."""
    drops: List[float] = []
    for _ in range(repeats):
        d = df_val.copy()
        for _, idx in d.groupby(list(BY_COLS)).groups.items():
            vals = d.loc[idx, feature].to_numpy()
            if len(vals) > 1:
                np.random.shuffle(vals)
                d.loc[idx, feature] = vals

        ranked = runner.rank(d, by=BY_COLS, include_probs=True, ascending=False)
        # выравниваем типы ключей
        for col in BY_COLS:
            if col in ranked.columns and np.issubdtype(ranked[col].dtype, np.floating):
                ranked[col] = ranked[col].astype("int32")

        joined = ranked.merge(
            d[list(KEY) + ["finish_pos_eff"]],
            on=list(KEY), how="left", suffixes=("", "_tgt"), validate="one_to_one"
        )
        fin_col = _pick_col(joined, "finish_pos_eff")
        if fin_col is None:
            return float("nan")

        s_col = _score_col(joined)
        if s_col is None:
            # фоллбэк: используем -rank как суррогат скора
            if "rank" in joined.columns:
                joined["__score_sur__"] = -pd.to_numeric(joined["rank"], errors="coerce")
                s_col = "__score_sur__"
            else:
                return float("nan")

        perf = -pd.to_numeric(joined[fin_col], errors="coerce")
        rho = _spearman(pd.to_numeric(joined[s_col], errors="coerce").to_numpy(), perf.to_numpy())
        drops.append(base_metric - (rho if np.isfinite(rho) else 0.0))
    return float(np.nanmean(drops)) if drops else float("nan")


# ---------- main audit ----------

def audit(
    artifacts_dir: str,
    features_pq: str,
    targets_pq: str,
    val_last: int = 6,
    out_csv: str = "audit_report.csv",
    perm_max_features: int = 40,
    perm_repeats: int = 3,
    target_hint: Optional[str] = None,
):
    # 1) загрузка
    feats = pd.read_parquet(features_pq)
    tgts = pd.read_parquet(targets_pq)

    # 2) сплит (и создание finish_pos_eff внутри)
    df_tr, df_va = _prepare_splits(feats, tgts, val_last=val_last, target_hint=target_hint)

    # 3) раннер и baseline ранк на валидации (просим вернуть score)
    runner = InferenceRunner.from_dir(artifacts_dir)
    base_rank = runner.rank(df_va.copy(), by=BY_COLS, include_probs=True, ascending=False)

    # ключи в int32 на всякий
    for col in BY_COLS:
        if col in base_rank.columns and np.issubdtype(base_rank[col].dtype, np.floating):
            base_rank[col] = base_rank[col].astype("int32")

    base_join = base_rank.merge(
        df_va[list(KEY) + ["finish_pos_eff"]],
        on=list(KEY), how="left", suffixes=("", "_tgt"), validate="one_to_one"
    )

    fin_col = _pick_col(base_join, "finish_pos_eff")
    if fin_col is None:
        raise KeyError(
            "finish_pos_eff is missing after merge — check that targets parquet contains a usable finish column. "
            f"Columns in joined DF: {list(base_join.columns)[:40]}"
        )

    s_col = _score_col(base_join)
    if s_col is None:
        # если раннер не вернул score — используем -rank
        if "rank" in base_join.columns:
            base_join["__score_sur__"] = -pd.to_numeric(base_join["rank"], errors="coerce")
            s_col = "__score_sur__"
        else:
            raise KeyError(
                "No score-like column returned by runner.rank (expected one of: score, pred, prediction, yhat, proba, logit, output)."
            )

    base_perf = -pd.to_numeric(base_join[fin_col], errors="coerce")
    base_rho = _spearman(pd.to_numeric(base_join[s_col], errors="coerce").to_numpy(), base_perf.to_numpy())

    # 4) список рабочих фич в нужном порядке
    feat_cols = list(getattr(runner, "feature_columns", getattr(runner.artifacts, "feature_cols", [])))
    if not feat_cols:
        # если совсем ничего — возьмём числовые из df_tr
        feat_cols = [c for c in df_tr.columns if c not in KEY and pd.api.types.is_numeric_dtype(df_tr[c])]

    # 5) grad*input важность (если доступен torch и трансформер)
    gradx_mean_abs = np.full(len(feat_cols), np.nan, dtype=np.float32)

    if _HAS_SCALER:
        try:
            import torch  # noqa: F401
            torch_available = True
        except Exception:
            torch_available = False
    else:
        torch_available = False

    if torch_available:
        import torch
        X_val = transform_with_scaler_df(base_join, feat_cols, runner.artifacts.scaler, as_array=True).astype(np.float32, copy=False)
        model = runner.artifacts.model.eval()
        xt = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            s = model(xt)  # [N]
        grads = []
        for i in range(xt.shape[0]):
            model.zero_grad(set_to_none=True)
            if xt.grad is not None:
                xt.grad.zero_()
            s[i].backward(retain_graph=True)
            grads.append(xt.grad[i].detach().cpu().numpy().copy())
        grad = np.vstack(grads)  # [N, F]
        contrib = np.abs(grad * xt.detach().cpu().numpy())
        gradx_mean_abs = contrib.mean(axis=0).astype(np.float32)

    # 6) коры по train/val (score и target)
    tr_rank = runner.rank(df_tr.copy(), by=BY_COLS, include_probs=True, ascending=False)
    for col in BY_COLS:
        if col in tr_rank.columns and np.issubdtype(tr_rank[col].dtype, np.floating):
            tr_rank[col] = tr_rank[col].astype("int32")

    tr_join = tr_rank.merge(
        df_tr[list(KEY) + ["finish_pos_eff"]],
        on=list(KEY), how="left", suffixes=("", "_tgt"), validate="one_to_one"
    )

    fin_tr_col = _pick_col(tr_join, "finish_pos_eff") or "finish_pos_eff"
    s_tr_col = _score_col(tr_join) or ("__score_sur__" if "rank" in tr_join.columns else None)
    if s_tr_col == "__score_sur__":
        tr_join[s_tr_col] = -pd.to_numeric(tr_join["rank"], errors="coerce")

    tr_perf = -pd.to_numeric(tr_join[fin_tr_col], errors="coerce")

    rows = []
    for j, col in enumerate(feat_cols):
        tr_col = pd.to_numeric(tr_join[col], errors="coerce")
        va_col = pd.to_numeric(base_join[col], errors="coerce")

        miss_tr = float(tr_col.isna().mean())
        miss_va = float(va_col.isna().mean())
        psi = _psi(tr_col, va_col, bins=10)

        rho_tr_score = _spearman(tr_col.to_numpy(), pd.to_numeric(tr_join[s_tr_col], errors="coerce").to_numpy()) if s_tr_col else float("nan")
        rho_va_score = _spearman(va_col.to_numpy(), pd.to_numeric(base_join[s_col], errors="coerce").to_numpy()) if s_col else float("nan")
        rho_tr_tgt = _spearman(tr_col.to_numpy(), tr_perf.to_numpy())
        rho_va_tgt = _spearman(va_col.to_numpy(), base_perf.to_numpy())

        exp = _expected_direction(col)
        pol_bad = None
        if exp is not None and np.isfinite(rho_va_score):
            pol_bad = int((exp > 0 and rho_va_score < 0) or (exp < 0 and rho_va_score > 0))

        rows.append(dict(
            feature=col,
            miss_train=miss_tr,
            miss_val=miss_va,
            psi=psi,
            rho_tr_score=rho_tr_score,
            rho_va_score=rho_va_score,
            rho_tr_target=rho_tr_tgt,
            rho_va_target=rho_va_tgt,
            exp_dir=exp,
            polarity_mismatch=pol_bad if pol_bad is not None else np.nan,
            gradx_mean_abs=float(gradx_mean_abs[j]) if np.isfinite(gradx_mean_abs[j]) else np.nan,
        ))

    rep = pd.DataFrame(rows)

    # 7) permutation importance по топ-N важных
    if rep.empty:
        raise RuntimeError("No features to audit. Check that feature columns match artifacts feature list.")

    if rep["gradx_mean_abs"].notna().any():
        cand = rep.sort_values("gradx_mean_abs", ascending=False).head(perm_max_features)["feature"].tolist()
    else:
        cand = (rep.assign(abs_rho=lambda d: d["rho_va_score"].abs())
                  .sort_values("abs_rho", ascending=False)
                  .head(perm_max_features)["feature"].tolist())

    perm_drops: Dict[str, float] = {}
    for f in cand:
        try:
            drop = _groupwise_permutation_drop(
                runner, df_va.copy(), base_rho, f, repeats=perm_repeats
            )
        except Exception as e:
            print(f"[perm] {f}: error {e}", file=sys.stderr)
            drop = np.nan
        perm_drops[f] = drop

    rep["perm_drop_spearman"] = rep["feature"].map(perm_drops).astype("float32")

    # 8) агрегат «подозрительности»
    flip_score = (np.sign(rep["rho_tr_score"].fillna(0)) != np.sign(rep["rho_va_score"].fillna(0))).astype(int)

    gnorm = rep["gradx_mean_abs"] / (rep["gradx_mean_abs"].max() + 1e-9)
    pdnorm = rep["perm_drop_spearman"].abs() / (rep["perm_drop_spearman"].abs().max() + 1e-9)

    rep["flip_score"] = flip_score
    rep["suspicious_score"] = (
        0.35 * rep["polarity_mismatch"].fillna(0).astype(float) +
        0.25 * rep["flip_score"].astype(float) +
        0.20 * (rep["psi"].fillna(0).clip(0, 2.0) / 2.0) +
        0.10 * gnorm.fillna(0) +
        0.10 * pdnorm.fillna(0)
    )

    rep = rep.sort_values(["suspicious_score", "perm_drop_spearman", "gradx_mean_abs"], ascending=False)

    # 9) сохранить и вывести топ
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_path, index=False)
    print(f"\nSaved audit report: {out_path}")

    print("\nTop 20 suspicious features:")
    cols = [
        "feature", "polarity_mismatch", "flip_score", "psi",
        "rho_va_score", "rho_va_target", "gradx_mean_abs",
        "perm_drop_spearman", "suspicious_score"
    ]
    cols = [c for c in cols if c in rep.columns]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(rep[cols].head(20).to_string(index=False))


def main():
    ap = argparse.ArgumentParser("Audit features behavior w.r.t model & target")
    ap.add_argument("--artifacts", required=True, help="path to trained artifacts dir")
    ap.add_argument("--features", required=True, help="path to all_features.parquet")
    ap.add_argument("--targets", required=True, help="path to all_targets.parquet")
    ap.add_argument("--val-last", type=int, default=6)
    ap.add_argument("--out-csv", default="audit_report.csv")
    ap.add_argument("--perm-max-features", type=int, default=40)
    ap.add_argument("--perm-repeats", type=int, default=3)
    ap.add_argument("--target-col", default=None, help="explicit target column name to derive finish_pos_eff (optional)")
    args = ap.parse_args()

    audit(
        artifacts_dir=args.artifacts,
        features_pq=args.features,
        targets_pq=args.targets,
        val_last=args.val_last,
        out_csv=args.out_csv,
        perm_max_features=args.perm_max_features,
        perm_repeats=args.perm_repeats,
        target_hint=args.target_col,
    )


if __name__ == "__main__":
    main()
