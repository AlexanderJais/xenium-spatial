"""
composition.py
--------------
Cell-type composition analysis: per-replicate proportions of each cluster /
cell type, compared across conditions.

Proportions are computed **per biological replicate** (one value per sample),
never per cell — treating thousands of cells as independent observations is
pseudoreplication and massively inflates significance. With the typical 2-vs-2
design the per-condition tests are underpowered, so this module is built for
effect sizes (log2 fold-change of mean proportion) with the test reported as an
exploratory ranking aid, clearly caveated in the UI.

Depends only on numpy/pandas/scipy (no scanpy) — it reads ``adata.obs``.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def composition_long(obs: pd.DataFrame, group_key: str = "cell_type",
                     sample_key: str = "replicate", condition_key: str = "condition",
                     batch_key: str = "batch") -> pd.DataFrame:
    """One row per (sample, group): ``count``, ``sample_total``, ``proportion``,
    plus condition/batch. Missing (sample, group) combinations are filled with a
    0 count so an absent cell type reads as 0 %, not as missing data.
    """
    if sample_key not in obs.columns or group_key not in obs.columns:
        raise KeyError(f"obs needs '{sample_key}' and '{group_key}' columns.")

    keep = [c for c in (sample_key, group_key, condition_key, batch_key) if c in obs.columns]
    df = obs[keep].copy()
    df[group_key] = df[group_key].astype(str)
    df[sample_key] = df[sample_key].astype(str)

    counts = (df.groupby([sample_key, group_key], observed=True).size()
                .rename("count").reset_index())

    # Complete the (sample × group) grid so zeros are explicit.
    samples = sorted(df[sample_key].unique())
    groups = sorted(df[group_key].unique())
    grid = pd.MultiIndex.from_product([samples, groups], names=[sample_key, group_key])
    counts = (counts.set_index([sample_key, group_key]).reindex(grid)
                    .reset_index())
    counts["count"] = counts["count"].fillna(0).astype(int)

    totals = counts.groupby(sample_key)["count"].sum()
    counts["sample_total"] = counts[sample_key].map(totals)
    counts["proportion"] = counts["count"] / counts["sample_total"].replace(0, np.nan)

    meta = df.drop_duplicates(sample_key).set_index(sample_key)
    for k in (condition_key, batch_key):
        if k in meta.columns:
            counts[k] = counts[sample_key].map(meta[k].astype(str))
    return counts


def composition_stats(comp_long: pd.DataFrame, group_key: str = "cell_type",
                      condition_key: str = "condition") -> pd.DataFrame:
    """Per group: per-condition mean proportion + replicate count, log2 fold
    change, and an exploratory Welch t-test (BH-adjusted) on the per-replicate
    proportions. With two conditions, fold change / test are the second vs the
    first alphabetically (e.g. **AGED vs ADULT**), so positive = higher in AGED.
    """
    from scipy import stats

    conds = sorted(comp_long[condition_key].dropna().unique()) \
        if condition_key in comp_long.columns else []
    # Fold-change pseudocount = half a cell as a proportion of the median sample
    # size. A fixed tiny value (e.g. 1e-9) is far below single-cell resolution,
    # so for a type absent in one condition it produces an absurd ±20+ log2FC; a
    # half-cell floor caps the change at the real detection limit.
    median_total = (float(comp_long["sample_total"].median())
                    if "sample_total" in comp_long.columns else 0.0)
    pc = 0.5 / median_total if median_total > 0 else 1e-3
    rows = []
    for g, sub in comp_long.groupby(group_key, observed=True):
        rec = {group_key: g}
        per_cond = []
        for c in conds:
            vals = sub.loc[sub[condition_key] == c, "proportion"].dropna().values
            rec[f"{c}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            rec[f"{c}_n"] = int(len(vals))
            per_cond.append(vals)
        if len(conds) == 2:
            a, b = rec[f"{conds[0]}_mean"], rec[f"{conds[1]}_mean"]
            rec["log2fc"] = float(np.log2((b + pc) / (a + pc)))
            rec["direction"] = f"{conds[1]} vs {conds[0]}"
            try:
                if all(len(v) >= 2 for v in per_cond):
                    rec["t_pval"] = float(
                        stats.ttest_ind(per_cond[1], per_cond[0], equal_var=False).pvalue)
                else:
                    rec["t_pval"] = np.nan
            except Exception:
                rec["t_pval"] = np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    # Always emit t_padj when a test was run (NaN where unpowered) so the column
    # contract is stable regardless of how many cell types were testable.
    if "t_pval" in out.columns:
        out["t_padj"] = _bh_adjust(out["t_pval"].to_numpy())
    sort_col = "log2fc" if "log2fc" in out.columns else group_key
    if sort_col == "log2fc":
        out = out.reindex(out["log2fc"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment, NaN-safe."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = pm.size
    if n == 0:
        return out
    order = np.argsort(pm)
    ranked = pm[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[mask] = res
    return out
