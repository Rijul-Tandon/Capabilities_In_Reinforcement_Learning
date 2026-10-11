"""
masking_diagnostics.py
======================
Diagnostics for "discretize continuous state with VQ-VAE, then penalize
actions that self-loop (z(s_t) == z(s_{t+1}))".

Everything here is numpy / sklearn / matplotlib only, so it runs on the
arrays you already have in the notebook.  Torch is only touched inside
`encoder_outputs()` (a convenience to pull z_e and the codebook out of your
VQVAEStateDiscretizer).

Minimal usage inside the notebook, after Cell 6 has run
-------------------------------------------------------
    import masking_diagnostics as md

    z_e, codebook = md.encoder_outputs(vqvae_model, states_norm, device)

    md.codebook_usage_report(states_disc, NUM_EMBEDDINGS)             # Q1: does one code eat everything?
    md.within_cluster_spread(states_norm.numpy(), states_disc, NUM_EMBEDDINGS)
    md.transition_matrix_plot(states_disc, next_states_disc, NUM_EMBEDDINGS)

    # Q2: which cells are actually action-separable (base-rate-free metrics)
    df_a = md.separability_report(actions_np, states_disc, is_self_loop,
                                  title="action only")

    # Q3: is "self-loop" really about the action, or about where s sits in the cell?
    md.boundary_margin_analysis(z_e, codebook, states_disc, is_self_loop, actions_np)
    df_ar = md.separability_report(actions_np, states_disc, is_self_loop,
                                   extra_features=md.cell_position_features(z_e, codebook, states_disc),
                                   title="action + position-in-cell")
    md.compare_reports({"action only": df_a, "action + position": df_ar})

    # Q4: do we need more codes?  (train a model per K in your own loop, then:)
    results = []
    for K, (codes, next_codes) in trained_models.items():
        results.append(md.evaluate_discretization(codes, next_codes, actions_np,
                                                  states_norm.numpy(), K))
    md.plot_k_sweep(results)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             matthews_corrcoef, precision_score, recall_score)

# --------------------------------------------------------------------------
# 0. Convenience: pull encoder latents + codebook out of the torch model
# --------------------------------------------------------------------------

def encoder_outputs(vqvae_model, states_norm, device="cpu", batch_size=8192):
    """Return (z_e [N,D] float32, codebook [K,D] float32) as numpy arrays.

    states_norm: torch tensor or numpy array of *normalized* states.
    """
    import torch  # local import so the rest of the module has no torch dependency
    vqvae_model.eval()
    if not torch.is_tensor(states_norm):
        states_norm = torch.tensor(np.asarray(states_norm), dtype=torch.float32)
    outs = []
    with torch.no_grad():
        for i in range(0, len(states_norm), batch_size):
            outs.append(vqvae_model.encoder(states_norm[i:i + batch_size].to(device)).cpu())
    z_e = torch.cat(outs).numpy().astype(np.float32)
    codebook = vqvae_model.vq_layer.embedding.weight.detach().cpu().numpy().astype(np.float32)
    return z_e, codebook


# --------------------------------------------------------------------------
# 1. Codebook usage: is one code swallowing the state space?
# --------------------------------------------------------------------------

def _usage_stats(codes: np.ndarray, K: int) -> dict:
    counts = np.bincount(codes, minlength=K).astype(float)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    perplexity = float(np.exp(-(nz * np.log(nz)).sum()))
    srt = np.sort(counts)[::-1]
    cum = np.cumsum(srt) / counts.sum()
    # Gini coefficient of usage (0 = perfectly uniform, 1 = one code has everything)
    n = len(counts)
    gini = float((2 * np.sum(np.arange(1, n + 1) * np.sort(counts)) / (n * counts.sum())) - (n + 1) / n)
    return dict(
        K=K,
        active_codes=int((counts > 0).sum()),
        perplexity=perplexity,
        perplexity_ratio=perplexity / K,
        max_share=float(probs.max()),
        top10_share=float(cum[min(9, n - 1)]),
        codes_for_50pct=int(np.searchsorted(cum, 0.5) + 1),
        codes_for_90pct=int(np.searchsorted(cum, 0.9) + 1),
        gini=gini,
        counts=counts,
    )


def codebook_usage_report(codes: np.ndarray, K: int, show=True):
    s = _usage_stats(codes, K)
    counts = s.pop("counts")
    print("=" * 70)
    print(f"CODEBOOK USAGE  (K = {K})")
    print("=" * 70)
    print(f"active codes            : {s['active_codes']} / {K}")
    print(f"perplexity              : {s['perplexity']:.1f}  ({100*s['perplexity_ratio']:.0f}% of K)")
    print(f"largest code share      : {100*s['max_share']:.2f}%   (uniform would be {100/K:.2f}%)")
    print(f"top-10 codes share      : {100*s['top10_share']:.1f}%")
    print(f"codes holding 50% data  : {s['codes_for_50pct']}")
    print(f"codes holding 90% data  : {s['codes_for_90pct']}")
    print(f"usage Gini              : {s['gini']:.2f}   (0 uniform, 1 collapsed)")
    if show:
        srt = np.sort(counts)[::-1]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120)
        ax[0].bar(np.arange(K), srt, width=1.0, color="tab:purple", alpha=.8)
        ax[0].axhline(len(codes) / K, color="k", ls="--", lw=1, label="uniform")
        ax[0].set_yscale("log"); ax[0].set_xlabel("codes (sorted by usage)"); ax[0].set_ylabel("samples")
        ax[0].set_title(f"Code usage, sorted  |  perplexity {s['perplexity']:.0f}/{K}"); ax[0].legend()
        cum = np.cumsum(srt) / srt.sum()
        ax[1].plot(np.arange(1, K + 1), cum, lw=2)
        ax[1].plot([0, K], [0, 1], "k--", lw=1, label="uniform")
        ax[1].axhline(.5, color="grey", lw=.8); ax[1].axhline(.9, color="grey", lw=.8)
        ax[1].set_xlabel("number of codes"); ax[1].set_ylabel("cumulative share of samples")
        ax[1].set_title("Concentration curve"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.show()
    return s


# --------------------------------------------------------------------------
# 2. Within-cluster spread: how "one-to-one" is state -> code really?
# --------------------------------------------------------------------------

def within_cluster_spread(states_norm: np.ndarray, codes: np.ndarray, K: int, show=True, verbose=True):
    """Per-code RMS distance of (normalized) states to their code's centroid.

    A VQ partition is many-to-one by construction; the meaningful question is
    whether the cells are *small and similar in size*.  Big spread in a big
    cell = "one cluster maps to many very different states".
    """
    states_norm = np.asarray(states_norm, dtype=np.float32)
    counts = np.bincount(codes, minlength=K)
    spread = np.full(K, np.nan)
    for k in np.where(counts > 1)[0]:
        x = states_norm[codes == k]
        spread[k] = np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean())
    global_rms = np.sqrt(((states_norm - states_norm.mean(0)) ** 2).sum(1).mean())
    ok = ~np.isnan(spread)
    if verbose:
        print("=" * 70)
        print("WITHIN-CELL SPREAD (RMS distance to cell centroid, normalized state units)")
        print("=" * 70)
        print(f"global RMS spread of data      : {global_rms:.3f}")
        print(f"median cell spread             : {np.nanmedian(spread):.3f}  "
              f"({100*np.nanmedian(spread)/global_rms:.0f}% of global)  "
              f"<- near 100% means codes ignore most state directions")
        print(f"sample-weighted mean spread    : {np.average(spread[ok], weights=counts[ok]):.3f}")
        print(f"max cell spread (code, n)      : {np.nanmax(spread):.3f}  "
              f"(code {int(np.nanargmax(spread))}, n={counts[int(np.nanargmax(spread))]})")
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120)
        ax[0].scatter(counts[ok], spread[ok], s=18, alpha=.7)
        ax[0].set_xscale("log"); ax[0].set_xlabel("cell size (samples)"); ax[0].set_ylabel("RMS spread")
        ax[0].axhline(global_rms, color="r", ls="--", lw=1, label="whole dataset")
        ax[0].set_title("Cell spread vs cell size"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].hist(spread[ok], bins=40, color="tab:orange", alpha=.8)
        ax[1].set_xlabel("RMS spread"); ax[1].set_ylabel("# cells"); ax[1].set_title("Distribution of cell spreads")
        plt.tight_layout(); plt.show()
    return pd.DataFrame({"code": np.arange(K), "n": counts, "spread": spread})


# --------------------------------------------------------------------------
# 3. Transition structure between codes
# --------------------------------------------------------------------------

def transition_matrix_plot(codes, next_codes, K, top=60, show=True):
    T = np.zeros((K, K))
    np.add.at(T, (codes, next_codes), 1)
    row = T.sum(1, keepdims=True); P = np.divide(T, row, out=np.zeros_like(T), where=row > 0)
    diag = np.diag(P)
    out_degree = (T > 0).sum(1)
    order = np.argsort(-row[:, 0])[:top]
    print("=" * 70)
    print("TRANSITION STRUCTURE")
    print("=" * 70)
    print(f"global self-loop rate           : {100*np.trace(T)/T.sum():.1f}%")
    print(f"median distinct successors/code : {np.median(out_degree[row[:,0]>0]):.0f}")
    print(f"codes with self-loop > 50%      : {(diag>0.5).sum()}   (>80%: {(diag>0.8).sum()})")
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        im = ax[0].imshow(P[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax[0].set_title(f"Row-normalized transitions, top-{top} codes by usage\n(diagonal = self-loop probability)")
        ax[0].set_xlabel("next code (sorted)"); ax[0].set_ylabel("code (sorted)")
        plt.colorbar(im, ax=ax[0], fraction=.046)
        ax[1].scatter(row[:, 0], diag, s=18, alpha=.7)
        ax[1].set_xscale("log"); ax[1].set_xlabel("cell size"); ax[1].set_ylabel("self-loop rate")
        ax[1].set_title("Self-loop rate vs cell size\n(bigger cells should self-loop more; check for the reverse)")
        ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.show()
    return P


# --------------------------------------------------------------------------
# 4. Separability that does NOT reward high base rates
# --------------------------------------------------------------------------

@dataclass
class ClusterSep:
    code: int
    n: int
    base_rate: float
    auroc: float
    bacc: float
    mcc: float
    precision: float
    recall: float
    lift: float          # precision / base_rate ; 1.0 == chance
    status: str


def _make_clf(kind: str):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.08,
                                              class_weight="balanced", random_state=0)
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=0)
    raise ValueError(kind)


def separability_report(actions: np.ndarray, codes: np.ndarray, is_self_loop: np.ndarray,
                        extra_features: Optional[np.ndarray] = None,
                        min_samples: int = 50, min_minority: int = 10,
                        classifier: str = "gbm", cv: int = 5,
                        auroc_threshold: float = 0.80, title: str = "",
                        show=True, verbose=True) -> pd.DataFrame:
    """Per-cell cross-validated separability of self-loop vs state-change.

    Uses AUROC (threshold-free, base-rate-free), balanced accuracy, MCC and the
    precision *lift* over base rate.  A cell whose precision merely equals its
    base rate has learned nothing -- lift ~ 1, AUROC ~ 0.5 -- even if that
    precision is 0.9.

    extra_features: optional [N, F] array concatenated to actions (e.g. the
    output of `cell_position_features`) to test what else predicts self-loops.
    """
    X_all = np.asarray(actions, dtype=np.float32)
    if extra_features is not None:
        X_all = np.concatenate([X_all, np.asarray(extra_features, dtype=np.float32)], axis=1)
    y_all = np.asarray(is_self_loop).astype(int)
    codes = np.asarray(codes)
    K = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=K)
    rows: List[ClusterSep] = []

    for k in np.argsort(-counts):
        n = counts[k]
        if n < min_samples:
            continue
        m = codes == k
        X, y = X_all[m], y_all[m]
        base = y.mean()
        n_min = min(y.sum(), n - y.sum())
        if n_min < min_minority:
            status = "ALL_SELF_LOOP" if base > .5 else "NO_SELF_LOOPS"
            rows.append(ClusterSep(k, n, base, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, status))
            continue
        folds = min(cv, n_min)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        proba = np.zeros(n); pred = np.zeros(n, dtype=int)
        for tr, te in skf.split(X, y):
            clf = _make_clf(classifier).fit(X[tr], y[tr])
            proba[te] = clf.predict_proba(X[te])[:, 1]
            pred[te] = (proba[te] >= 0.5).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auroc = roc_auc_score(y, proba)
            bacc = balanced_accuracy_score(y, pred)
            mcc = matthews_corrcoef(y, pred)
            prec = precision_score(y, pred, zero_division=0)
            rec = recall_score(y, pred, zero_division=0)
        status = "SEPARABLE" if auroc >= auroc_threshold else "NOT_SEPARABLE"
        rows.append(ClusterSep(k, n, base, auroc, bacc, mcc, prec, rec, prec / max(base, 1e-9), status))

    df = pd.DataFrame([asdict(r) for r in rows])
    ok = df["auroc"].notna()
    w = df.loc[ok, "n"]
    maskable = df.loc[df.status == "SEPARABLE", "n"].sum() / len(codes)
    if verbose:
        print("=" * 78)
        print(f"SEPARABILITY REPORT {('— ' + title) if title else ''}  "
              f"[{classifier}, {cv}-fold CV, features dim = {X_all.shape[1]}]")
        print("=" * 78)
        print(f"cells evaluated                       : {len(df)}  "
              f"(degenerate: {(~ok).sum()})")
        print(f"sample-weighted AUROC                 : {np.average(df.loc[ok,'auroc'], weights=w):.3f}")
        print(f"sample-weighted balanced accuracy     : {np.average(df.loc[ok,'bacc'], weights=w):.3f}")
        print(f"sample-weighted MCC                   : {np.average(df.loc[ok,'mcc'],  weights=w):.3f}")
        print(f"cells with AUROC >= {auroc_threshold:.2f}               : {(df.status=='SEPARABLE').sum()}")
        print(f"share of transitions in those cells   : {100*maskable:.1f}%   <- this is your usable mask coverage")
        print(f"corr(precision, base rate)            : {np.corrcoef(df.loc[ok,'precision'], df.loc[ok,'base_rate'])[0,1]:.2f}")
        print(f"corr(AUROC, base rate)                : {np.corrcoef(df.loc[ok,'auroc'],     df.loc[ok,'base_rate'])[0,1]:.2f}")
    if show:
        _plot_separability(df, auroc_threshold, title)
    return df


def _plot_separability(df: pd.DataFrame, thr: float, title: str):
    d = df[df["auroc"].notna()]
    sz = 12 + 220 * np.sqrt(d["n"] / d["n"].max())
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), dpi=120)
    sc = ax[0].scatter(d.base_rate, d.auroc, s=sz, c=d.base_rate, cmap="viridis", alpha=.7, edgecolor="k", lw=.3)
    ax[0].axhline(.5, color="k", ls="--", lw=1); ax[0].axhline(thr, color="crimson", ls=":", lw=1.2, label=f"threshold {thr}")
    ax[0].set_xlabel("self-loop base rate"); ax[0].set_ylabel("CV AUROC"); ax[0].set_title("Separability vs base rate\n(size = cell samples)")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_ylim(.35, 1.02)

    ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="precision = base rate")
    ax[1].scatter(d.base_rate, d.precision, s=sz, c=d.auroc, cmap="RdYlGn", vmin=.5, vmax=1, alpha=.8, edgecolor="k", lw=.3)
    ax[1].set_xlabel("self-loop base rate"); ax[1].set_ylabel("precision (self-loop class)")
    ax[1].set_title("Why raw precision is misleading\n(color = AUROC)"); ax[1].legend(); ax[1].grid(alpha=.3)

    order = np.argsort(-d.auroc.values); cum = np.cumsum(d.n.values[order]) / df.n.sum()
    ax[2].plot(d.auroc.values[order], cum, lw=2)
    ax[2].axvline(thr, color="crimson", ls=":", lw=1.2)
    ax[2].set_xlabel("AUROC threshold"); ax[2].set_ylabel("share of transitions in cells above threshold")
    ax[2].set_title("Mask coverage vs. how strict you are"); ax[2].grid(alpha=.3); ax[2].invert_xaxis()
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


def compare_reports(reports: Dict[str, pd.DataFrame], show=True):
    """Side-by-side of several separability_report outputs (same cells)."""
    keys = list(reports)
    base = reports[keys[0]][["code", "n"]]
    merged = base.copy()
    for k in keys:
        merged[k] = reports[k].set_index("code").loc[merged.code, "auroc"].values
    ok = merged[keys].notna().all(1)
    print("=" * 70); print("FEATURE-SET COMPARISON (sample-weighted AUROC)"); print("=" * 70)
    for k in keys:
        print(f"{k:<32}: {np.average(merged.loc[ok,k], weights=merged.loc[ok,'n']):.3f}")
    if show and len(keys) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
        sz = 12 + 220 * np.sqrt(merged.loc[ok, "n"] / merged.n.max())
        ax.scatter(merged.loc[ok, keys[0]], merged.loc[ok, keys[1]], s=sz, alpha=.7, edgecolor="k", lw=.3)
        ax.plot([.4, 1], [.4, 1], "k--", lw=1)
        ax.set_xlabel(f"AUROC: {keys[0]}"); ax.set_ylabel(f"AUROC: {keys[1]}")
        ax.set_title("Per-cell gain from extra features\n(points above the diagonal = self-loop was not\n"
                     "an action-only property in that cell)", fontsize=10)
        ax.grid(alpha=.3); plt.tight_layout(); plt.show()
    return merged


# --------------------------------------------------------------------------
# 5. Position-in-cell: the confound behind "unseparable" cells
# --------------------------------------------------------------------------

def _dists_to_codebook(z_e, codebook, chunk=20000):
    d = np.empty((len(z_e), len(codebook)), dtype=np.float32)
    cb2 = (codebook ** 2).sum(1)
    for i in range(0, len(z_e), chunk):
        z = z_e[i:i + chunk]
        d[i:i + chunk] = np.sqrt(np.maximum((z ** 2).sum(1, keepdims=True) + cb2 - 2 * z @ codebook.T, 0))
    return d


def cell_position_features(z_e, codebook, codes):
    """[N, D+2] features describing where s sits inside its Voronoi cell:
    residual vector (z_e - e_k), distance to own code, and margin to the
    second-nearest code.  Feed as `extra_features` to separability_report."""
    z_e = np.asarray(z_e, np.float32); codebook = np.asarray(codebook, np.float32)
    d = _dists_to_codebook(z_e, codebook)
    d_sorted = np.sort(d, axis=1)
    d1, d2 = d_sorted[:, 0], d_sorted[:, 1]
    resid = z_e - codebook[codes]
    return np.concatenate([resid, d1[:, None], (d2 - d1)[:, None]], axis=1)


def boundary_margin_analysis(z_e, codebook, codes, is_self_loop, actions, bins=10, show=True):
    """Does self-looping depend on the action, on the distance to the cell
    boundary, or both?  If the margin axis explains most of the variance,
    an action-only mask M(k, a) cannot work: the same action self-loops from
    the cell centre and exits from the cell edge."""
    z_e = np.asarray(z_e, np.float32); codebook = np.asarray(codebook, np.float32)
    y = np.asarray(is_self_loop).astype(float)
    d = _dists_to_codebook(z_e, codebook); ds = np.sort(d, 1)
    margin = ds[:, 1] - ds[:, 0]
    a_norm = np.linalg.norm(np.asarray(actions, np.float32), axis=1)

    def binned(x):
        edges = np.quantile(x, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, x, "right") - 1, 0, bins - 1)
        rate = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
        centers = 0.5 * (edges[1:] + edges[:-1]); return centers, rate, idx

    mc, mr, mi = binned(margin); ac, ar, ai = binned(a_norm)
    grid = np.full((bins, bins), np.nan)
    for i in range(bins):
        for j in range(bins):
            m = (mi == i) & (ai == j)
            if m.sum() >= 30: grid[i, j] = y[m].mean()
    # how much each axis explains (one-way ANOVA style R^2 on binned means)
    def r2(idx):
        means = np.array([y[idx == b].mean() for b in range(bins)])
        return 1 - ((y - means[idx]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("=" * 70); print("BOUNDARY-MARGIN vs ACTION-MAGNITUDE"); print("=" * 70)
    print(f"variance of self-loop explained by margin deciles   : {100*r2(mi):.1f}%")
    print(f"variance of self-loop explained by |a| deciles      : {100*r2(ai):.1f}%")
    print(f"self-loop rate, closest-to-boundary decile -> deepest-in-cell decile: "
          f"{100*mr[0]:.0f}% -> {100*mr[-1]:.0f}%")
    print(f"self-loop rate, smallest |a| decile -> largest |a| decile           : "
          f"{100*ar[0]:.0f}% -> {100*ar[-1]:.0f}%")
    if show:
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), dpi=120)
        ax[0].plot(mc, mr, "o-", lw=2); ax[0].set_xlabel("margin to 2nd-nearest code  (0 = on the boundary)")
        ax[0].set_ylabel("self-loop rate"); ax[0].set_title("Self-loop vs depth inside cell"); ax[0].grid(alpha=.3)
        ax[1].plot(ac, ar, "o-", lw=2, color="tab:red"); ax[1].set_xlabel("||a||"); ax[1].set_ylabel("self-loop rate")
        ax[1].set_title("Self-loop vs action magnitude"); ax[1].grid(alpha=.3)
        im = ax[2].imshow(grid, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(grid))
        ax[2].set_xlabel("||a|| decile (small -> large)"); ax[2].set_ylabel("margin decile (edge -> centre)")
        ax[2].set_title("Self-loop rate: margin x action"); plt.colorbar(im, ax=ax[2], fraction=.046)
        plt.tight_layout(); plt.show()
    return dict(margin=margin, a_norm=a_norm, r2_margin=r2(mi), r2_action=r2(ai))


# --------------------------------------------------------------------------
# 6. K sweep: one row of summary numbers per trained discretizer
# --------------------------------------------------------------------------

def evaluate_discretization(codes, next_codes, actions, states_norm, K,
                            auroc_threshold=0.80, classifier="gbm", cv=3,
                            max_cells_for_sep=None, extra_features=None) -> dict:
    """Everything you need to compare discretizers with different K (or
    different training recipes) on one line each.  Cheap enough to run in a
    loop; set max_cells_for_sep to only score the N largest cells if slow."""
    codes = np.asarray(codes); next_codes = np.asarray(next_codes)
    is_sl = codes == next_codes
    u = _usage_stats(codes, K); u.pop("counts")
    sp = within_cluster_spread(states_norm, codes, K, show=False, verbose=False)
    counts = sp.n.values; ok = sp.spread.notna()
    if max_cells_for_sep is not None:
        keep = np.isin(codes, np.argsort(-counts)[:max_cells_for_sep])
    else:
        keep = np.ones(len(codes), bool)
    ef_keep = extra_features[keep] if extra_features is not None else None
    df = separability_report(actions[keep], codes[keep], is_sl[keep], extra_features=ef_keep, classifier=classifier, cv=cv,
                             auroc_threshold=auroc_threshold, show=False, verbose=False)
    d = df[df.auroc.notna()]
    return dict(
        **u,
        self_loop_rate=float(is_sl.mean()),
        weighted_spread=float(np.average(sp.spread[ok], weights=counts[ok])),
        weighted_auroc=float(np.average(d.auroc, weights=d.n)) if len(d) else np.nan,
        weighted_bacc=float(np.average(d.bacc, weights=d.n)) if len(d) else np.nan,
        maskable_share=float(df.loc[df.status == "SEPARABLE", "n"].sum() / keep.sum()),
        n_separable=int((df.status == "SEPARABLE").sum()),
    )


def plot_k_sweep(results: Iterable[dict]):
    df = pd.DataFrame(list(results)).sort_values("K")
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), dpi=120)
    ax = ax.ravel()
    def p(i, col, title, ylim=None, ref=None):
        ax[i].plot(df.K, df[col], "o-", lw=2); ax[i].set_xscale("log", base=2); ax[i].set_xlabel("K")
        ax[i].set_title(title); ax[i].grid(alpha=.3)
        if ylim: ax[i].set_ylim(*ylim)
        if ref is not None: ax[i].axhline(ref, color="k", ls="--", lw=1)
    p(0, "perplexity_ratio", "codebook utilisation (perplexity / K)", (0, 1.05), 1)
    p(1, "max_share", "largest cell share", (0, None))
    p(2, "weighted_spread", "within-cell spread (sample-weighted)", (0, None))
    p(3, "self_loop_rate", "global self-loop rate", (0, 1))
    p(4, "weighted_auroc", "action-only separability (weighted AUROC)", (.45, 1.02), .5)
    p(5, "maskable_share", "share of transitions in separable cells", (0, 1))
    fig.suptitle("Discretizer sweep: you want utilisation & separability up,\n"
                 "spread down, and a self-loop rate that is still non-trivial", fontsize=12)
    plt.tight_layout(); plt.show()
    return df


# --------------------------------------------------------------------------
# 7. A readable replacement for the 256-colour PCA scatter
# --------------------------------------------------------------------------

def pca_cells_plot(states_np, codes, sep_df: Optional[pd.DataFrame] = None, K=None, show=True):
    """PCA of states with one marker per *cell centroid* (size = cell size,
    colour = self-loop rate or AUROC) instead of 200k points in 256 colours."""
    X = np.asarray(states_np, np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    U, S, Vt = np.linalg.svd(X, full_matrices=False); P = X @ Vt[:2].T
    K = K or int(codes.max()) + 1
    counts = np.bincount(codes, minlength=K)
    cent = np.array([P[codes == k].mean(0) if counts[k] else [np.nan, np.nan] for k in range(K)])
    spread = np.array([np.sqrt(((P[codes == k] - cent[k]) ** 2).sum(1).mean()) if counts[k] > 1 else np.nan for k in range(K)])
    color = None; label = "cell size"
    if sep_df is not None:
        s = sep_df.set_index("code"); color = np.full(K, np.nan)
        color[s.index] = s["auroc"].values; label = "action-only AUROC"
    if show:
        fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
        ax.scatter(P[::20, 0], P[::20, 1], s=2, c="lightgrey", alpha=.4)
        sc = ax.scatter(cent[:, 0], cent[:, 1], s=8 + 900 * counts / counts.max(),
                        c=color if color is not None else counts, cmap="RdYlGn" if color is not None else "viridis",
                        vmin=.5 if color is not None else None, vmax=1 if color is not None else None,
                        alpha=.85, edgecolor="k", lw=.4)
        for k in np.argsort(-counts)[:5]:
            ax.annotate(str(k), cent[k], fontsize=8, ha="center", va="center", fontweight="bold")
        plt.colorbar(sc, label=label)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Cell centroids in state PCA  (size = cell size)")
        plt.tight_layout(); plt.show()
    return pd.DataFrame({"code": np.arange(K), "n": counts, "pc1": cent[:, 0], "pc2": cent[:, 1], "pca_spread": spread})
