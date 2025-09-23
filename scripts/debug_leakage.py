# scripts/debug_leakage.py
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path

FEAT_PATH = "data/processed/all_features.parquet"
TGT_PATH  = "data/processed/all_targets.parquet"
VAL_LAST  = 6  # как в твоём запуске

# — ключи и имя таргета: поправь при необходимости
KEYS = ["raceId", "driverId"]
TARGET_COL = "finish_pos"  # или твой целевой столбец ранжирования

# шаблоны, которые почти всегда означают утечку
SUSPECT_NAME_PATTERNS = [
    "finish", "result", "position", "pos_", "_pos", "rank", "order",
    "points", "classified", "status", "laps_completed", "pitstops_total",
    "penalty", "delta_to_*", "gap_to_*", "post", "_actual", "_final"
]

def _looks_suspect(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in SUSPECT_NAME_PATTERNS)

def ndcg_at_k(sorted_truth, k=5):
    # упрощённый NDCG: ideal == 1.0 при идеальном порядке (меньше позиция — лучше)
    gains = 1.0 / np.log2(np.arange(2, len(sorted_truth)+2))
    ideal = gains[:k].sum()
    return gains[:k].sum() / ideal if ideal > 0 else 0.0

def main():
    dfX = pd.read_parquet(FEAT_PATH)
    dfY = pd.read_parquet(TGT_PATH)

    # убедимся, что таргет один столбец
    if TARGET_COL not in dfY.columns:
        # попробуем угадать
        candidates = [c for c in dfY.columns if c.lower() in ["finish_pos","finish_position","result_pos","rank"]]
        assert candidates, f"Не найден TARGET_COL='{TARGET_COL}'. Доступны: {dfY.columns.tolist()}"
        target = candidates[0]
    else:
        target = TARGET_COL

    df = dfX.merge(dfY[KEYS+[target]], on=KEYS, how="inner")
    df = df.sort_values(["raceId"]).reset_index(drop=True)

    # сплитация как в тренере: последние VAL_LAST гонок — в валидацию
    races = df["raceId"].drop_duplicates().tolist()
    val_races = set(races[-VAL_LAST:])
    df_val = df[df["raceId"].isin(val_races)].copy()

    # БЫСТРАЯ САНИТАРКА
    # 1) полностью NaN / константы → это ломает нормализацию
    num_cols = [c for c in dfX.columns if c not in KEYS and pd.api.types.is_numeric_dtype(dfX[c])]
    all_nan = [c for c in num_cols if dfX[c].isna().all()]
    constants = []
    for c in num_cols:
        x = dfX[c].dropna().values
        if x.size and np.nanstd(x) == 0:
            constants.append(c)

    # 2) точное совпадение с таргетом на валидации (или с -таргетом)
    exact_eq, exact_neg_eq = [], []
    yv = df_val[target].to_numpy()
    for c in num_cols:
        xv = df_val[c].to_numpy()
        if xv.size != yv.size: 
            continue
        if np.allclose(xv, yv, equal_nan=False):
            exact_eq.append(c)
        if np.allclose(-xv, yv, equal_nan=False):
            exact_neg_eq.append(c)

    # 3) Спирмен по гонкам: если в КАЖДОЙ вал-гонке |rho|>0.999 → очень подозрительно
    suspicious_corr = []
    by_race = dict(tuple(df_val.groupby("raceId")))
    for c in num_cols:
        # пропустим очевидно мусорное
        if c in all_nan: 
            continue
        ok = True
        rhos = []
        for rid, g in by_race.items():
            x = g[c].to_numpy()
            y = g[target].to_numpy()
            if np.all(np.isnan(x)) or np.nanstd(x) == 0:
                ok = False
                break
            # спирмен устойчив к монотонным преобразованиям
            rho, _ = spearmanr(x, y, nan_policy="omit")
            if np.isnan(rho):
                ok = False
                break
            rhos.append(abs(rho))
        if ok and rhos and min(rhos) > 0.999:
            suspicious_corr.append((c, float(np.mean(rhos)), min(rhos)))

    suspicious_corr.sort(key=lambda t: -t[1])

    # 4) Эвристика по именам
    name_flagged = [c for c in num_cols if _looks_suspect(c)]

    # 5) NDCG@5 при простом сортинге по фиче (если уже 1.0 — прямая утечка порядка)
    ndcg_hits = []
    for c in num_cols:
        all_good = True
        scores = []
        for rid, g in by_race.items():
            g = g.dropna(subset=[c])
            if g.empty: 
                continue
            # если фича — "лучше меньше", сравни обе ориентации и возьми лучшую
            ord1 = g.sort_values(c, ascending=True)[target].values
            ord2 = g.sort_values(c, ascending=False)[target].values
            s = max(ndcg_at_k(ord1, 5), ndcg_at_k(ord2, 5))
            scores.append(s)
        if scores and min(scores) > 0.999:
            ndcg_hits.append((c, float(np.mean(scores)), min(scores)))
    ndcg_hits.sort(key=lambda t: -t[1])

    print("\n=== Полностью NaN колонки (drop):", len(all_nan))
    print(all_nan[:50])
    print("\n=== Константы (drop):", len(constants))
    print(constants[:50])

    print("\n=== Точное равенство таргету на валидации:", exact_eq)
    print("=== Точное равенство -таргету на валидации:", exact_neg_eq)

    print("\n=== Очень высокая Спирмен-связь по всем вал-гонкам (rho>0.999):")
    for c, mean_rho, min_rho in suspicious_corr[:30]:
        print(f"{c:40s}  mean_rho={mean_rho:.5f}  min_rho={min_rho:.5f}")

    print("\n=== NDCG@5==1.0 при сортировке по фиче (везде):")
    for c, mean_s, min_s in ndcg_hits[:30]:
        print(f"{c:40s}  mean_ndcg={mean_s:.3f}  min_ndcg={min_s:.3f}")

    print("\n=== По имени выглядят подозрительно:")
    print(name_flagged[:80])

    # Сохранить списки для ручного дропа
    outdir = Path("debug_out"); outdir.mkdir(exist_ok=True)
    pd.Series(all_nan).to_csv(outdir/"all_nan_cols.csv", index=False)
    pd.Series(constants).to_csv(outdir/"constant_cols.csv", index=False)
    pd.DataFrame(suspicious_corr, columns=["col","mean_rho","min_rho"]).to_csv(outdir/"suspicious_by_corr.csv", index=False)
    pd.DataFrame(ndcg_hits, columns=["col","mean_ndcg","min_ndcg"]).to_csv(outdir/"suspicious_by_ndcg.csv", index=False)
    pd.Series(name_flagged).to_csv(outdir/"name_flagged.csv", index=False)

if __name__ == "__main__":
    main()
