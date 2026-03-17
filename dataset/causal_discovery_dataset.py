"""
Data-Driven Causal Discovery on the Merged EBLNN Dataset
==========================================================
Verifies whether structural causality exists in the merged dataset
(real furnace data + synthetic edge cases) WITHOUT any model training.

Three complementary methods:

1. Partial Correlation DAG (constraint-based, like the PC algorithm skeleton)
   - Removes spurious correlations by conditioning on other variables
   - Edge = statistically significant partial correlation after FDR correction

2. Granger Causality (time-lagged, but applied cross-sectionally)
   - Tests if knowing X improves prediction of Y beyond Y alone
   - Uses VAR-style lagged regression on the dataset

3. Transfer Entropy Approximation (nonlinear, information-theoretic)
   - Bins continuous variables and measures directed information transfer
   - Captures nonlinear dependencies that partial correlation misses

Ground-truth expectations from furnace physics:
   fuel_flow  → current_temp    (energy input heats process fluid)
   fuel_flow  → next_temp       (directly via energy balance)
   air_fuel_ratio → next_excess_o2  (mass balance: excess air = excess O2)
   inflow_temp → next_temp      (inlet temperature passes through)
   inflow_rate → next_temp      (process load: more mass = less ΔT)
   fuel_flow ↔ air_fuel_ratio   (coupled in combustion)

Usage:
    python causal_discovery_dataset.py
    python causal_discovery_dataset.py --data ../dataset/merged_eblnn_dataset.csv
    python causal_discovery_dataset.py --n_samples 50000  # subsample for speed
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from itertools import combinations

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ── Feature and target labels ──────────────────────────────────────────
INPUT_COLS  = ["fuel_flow", "air_fuel_ratio", "current_temp",
               "inflow_temp", "inflow_rate"]
TARGET_COLS = ["next_temp", "next_excess_o2"]
ALL_COLS    = INPUT_COLS + TARGET_COLS

# Short labels for plots
SHORT = {
    "fuel_flow": "fuel",
    "air_fuel_ratio": "AFR",
    "current_temp": "cur_T",
    "inflow_temp": "in_T",
    "inflow_rate": "in_rate",
    "next_temp": "nxt_T",
    "next_excess_o2": "nxt_O2",
}

# ── Domain-knowledge ground truth edges ────────────────────────────────
GROUND_TRUTH_EDGES = [
    ("fuel_flow",       "next_temp"),
    ("fuel_flow",       "current_temp"),
    ("air_fuel_ratio",  "next_excess_o2"),
    ("inflow_temp",     "next_temp"),
    ("inflow_rate",     "next_temp"),
    ("current_temp",    "next_temp"),
    ("fuel_flow",       "air_fuel_ratio"),  # coupled
]


# ======================================================================
# 1. Partial Correlation Matrix
# ======================================================================
def partial_corr_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute the partial correlation between every pair (i, j),
    conditioning on ALL other variables.

    Uses the inverse of the correlation matrix (precision matrix):
        pcorr(i,j) = -P_ij / sqrt(P_ii * P_jj)
    """
    corr = df[cols].corr().values
    try:
        P = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(corr)

    d = len(cols)
    pcorr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                pcorr[i, j] = 1.0
            else:
                pcorr[i, j] = -P[i, j] / np.sqrt(P[i, i] * P[j, j] + 1e-12)
    return pd.DataFrame(pcorr, index=cols, columns=cols)


def partial_corr_significance(pcorr_val: float, n: int, k: int) -> float:
    """
    Two-sided p-value for partial correlation.
    n = number of observations, k = number of conditioning variables.
    t = pcorr * sqrt((n - k - 2) / (1 - pcorr^2))
    """
    dof = n - k - 2
    if dof <= 0 or abs(pcorr_val) >= 1.0:
        return 0.0
    t_stat = pcorr_val * np.sqrt(dof / (1.0 - pcorr_val**2 + 1e-12))
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), dof)
    return p_val


def partial_corr_dag(df: pd.DataFrame, cols: list, alpha: float = 0.01):
    """
    Build an undirected skeleton from significant partial correlations.
    Returns the partial correlation matrix, p-value matrix, and adjacency.
    """
    pcorr = partial_corr_matrix(df, cols)
    n = len(df)
    k = len(cols) - 2  # conditioning set size
    d = len(cols)

    pvals = np.ones((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            p = partial_corr_significance(pcorr.values[i, j], n, k)
            pvals[i, j] = p
            pvals[j, i] = p

    # FDR correction (Benjamini-Hochberg)
    upper_tri = [(i, j) for i in range(d) for j in range(i + 1, d)]
    raw_p = [pvals[i, j] for i, j in upper_tri]
    m = len(raw_p)
    sorted_idx = np.argsort(raw_p)
    adjusted = np.ones(m)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = raw_p[idx] * m / (rank + 1)
    adjusted = np.minimum.accumulate(adjusted[np.argsort(sorted_idx)][::-1])[::-1]

    adj = np.zeros((d, d))
    for idx_pair, (i, j) in enumerate(upper_tri):
        if adjusted[idx_pair] < alpha:
            adj[i, j] = abs(pcorr.values[i, j])
            adj[j, i] = abs(pcorr.values[i, j])

    return pcorr, pd.DataFrame(pvals, index=cols, columns=cols), adj


# ======================================================================
# 2. Granger-style Causality (cross-sectional via conditional regression)
# ======================================================================
def granger_cross_sectional(df: pd.DataFrame, cols: list) -> np.ndarray:
    """
    For each directed pair (X → Y), test whether X adds predictive power
    for Y beyond all other variables.

    Restricted model: Y ~ all cols except X and Y
    Full model:       Y ~ all cols except Y

    F-test on the improvement. Returns (d, d) matrix of -log10(p-values).
    """
    from sklearn.linear_model import LinearRegression

    d = len(cols)
    n = len(df)
    F_mat = np.zeros((d, d))

    for j in range(d):  # target
        y = df[cols[j]].values
        other_cols = [c for c in cols if c != cols[j]]

        for i in range(d):  # candidate cause
            if i == j:
                continue

            # Restricted: predict Y without X_i
            restricted_cols = [c for c in other_cols if c != cols[i]]
            if len(restricted_cols) == 0:
                continue
            X_r = df[restricted_cols].values
            X_f = df[other_cols].values

            reg_r = LinearRegression().fit(X_r, y)
            reg_f = LinearRegression().fit(X_f, y)

            rss_r = np.sum((y - reg_r.predict(X_r))**2)
            rss_f = np.sum((y - reg_f.predict(X_f))**2)

            p_r = len(restricted_cols)
            p_f = len(other_cols)
            df1 = p_f - p_r  # should be 1
            df2 = n - p_f - 1

            if rss_f < 1e-12 or df2 <= 0:
                F_mat[i, j] = 0
                continue

            F_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
            p_val = stats.f.sf(F_stat, df1, df2)
            F_mat[i, j] = -np.log10(p_val + 1e-300)  # higher = more significant

    return F_mat


# ======================================================================
# 3. Transfer Entropy Approximation (binned, nonlinear)
# ======================================================================
def binned_transfer_entropy(df: pd.DataFrame, cols: list,
                            n_bins: int = 10) -> np.ndarray:
    """
    Approximate transfer entropy TE(X → Y) using binned distributions.

    TE(X→Y) = H(Y|Y_past) - H(Y|Y_past, X)

    Since we don't have explicit time series here, we use:
    TE(X→Y) ≈ MI(X; Y | Z) where Z = all other variables
    Approximated via: H(Y|Z) - H(Y|Z,X)

    Binning converts continuous variables to discrete for entropy estimation.
    """
    d = len(cols)
    te = np.zeros((d, d))

    # Bin all columns
    binned = pd.DataFrame()
    for col in cols:
        binned[col] = pd.qcut(df[col], q=n_bins, labels=False,
                               duplicates="drop")

    for i in range(d):  # source X
        for j in range(d):  # target Y
            if i == j:
                continue

            x_col = cols[i]
            y_col = cols[j]
            z_cols = [c for c in cols if c not in (x_col, y_col)]

            # H(Y | Z) via conditional entropy
            h_y_given_z = _cond_entropy(binned, y_col, z_cols)
            # H(Y | Z, X)
            h_y_given_zx = _cond_entropy(binned, y_col, z_cols + [x_col])

            te[i, j] = max(0, h_y_given_z - h_y_given_zx)

    return te


def _cond_entropy(binned: pd.DataFrame, target: str, cond_cols: list) -> float:
    """H(target | cond_cols) via groupby counting."""
    if len(cond_cols) == 0:
        # Just marginal entropy
        p = binned[target].value_counts(normalize=True).values
        return -np.sum(p * np.log2(p + 1e-12))

    # Group by conditioning variables
    # Use a hash of conditions for speed
    grouped = binned.groupby(cond_cols, observed=True)
    n_total = len(binned)
    h = 0.0
    for _, group in grouped:
        p_z = len(group) / n_total
        p_y_given_z = group[target].value_counts(normalize=True).values
        h += p_z * (-np.sum(p_y_given_z * np.log2(p_y_given_z + 1e-12)))
    return h


# ======================================================================
# Visualisation
# ======================================================================
def plot_partial_corr(pcorr: pd.DataFrame, adj: np.ndarray,
                      cols: list, out_dir: Path):
    """Heatmap of partial correlations with significant edges highlighted."""
    short_labels = [SHORT[c] for c in cols]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: partial correlation
    ax = axes[0]
    mask = np.eye(len(cols), dtype=bool)
    sns.heatmap(pcorr.values, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".3f",
                xticklabels=short_labels, yticklabels=short_labels,
                mask=mask, square=True, cbar_kws={"label": "Partial Correlation"})
    ax.set_title("Partial Correlation Matrix\n(conditioned on all other variables)")

    # Right: significant edges (adjacency)
    ax = axes[1]
    sns.heatmap(adj, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                annot=True, fmt=".3f",
                xticklabels=short_labels, yticklabels=short_labels,
                square=True, cbar_kws={"label": "|Partial Corr| (FDR < 0.01)"})
    ax.set_title("Significant Partial Correlations\n(FDR-corrected, α = 0.01)")

    plt.tight_layout()
    path = out_dir / "partial_correlation_dag.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_granger(F_mat: np.ndarray, cols: list, out_dir: Path):
    """Heatmap of Granger F-test significance (-log10 p)."""
    short_labels = [SHORT[c] for c in cols]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    mask = np.eye(len(cols), dtype=bool)
    sns.heatmap(F_mat, ax=ax, cmap="YlOrRd", vmin=0,
                annot=True, fmt=".1f",
                xticklabels=short_labels, yticklabels=short_labels,
                mask=mask, square=True,
                cbar_kws={"label": "-log₁₀(p-value)"})
    ax.set_title("Cross-Sectional Granger Causality\n(row → col, higher = stronger)")
    ax.set_xlabel("Target (effect)")
    ax.set_ylabel("Source (cause)")

    plt.tight_layout()
    path = out_dir / "granger_causality.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_transfer_entropy(te: np.ndarray, cols: list, out_dir: Path):
    """Heatmap of transfer entropy estimates."""
    short_labels = [SHORT[c] for c in cols]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    mask = np.eye(len(cols), dtype=bool)
    sns.heatmap(te, ax=ax, cmap="YlGnBu", vmin=0,
                annot=True, fmt=".4f",
                xticklabels=short_labels, yticklabels=short_labels,
                mask=mask, square=True,
                cbar_kws={"label": "Transfer Entropy (bits)"})
    ax.set_title("Transfer Entropy (Nonlinear Causality)\n(row → col, higher = more information transfer)")
    ax.set_xlabel("Target (effect)")
    ax.set_ylabel("Source (cause)")

    plt.tight_layout()
    path = out_dir / "transfer_entropy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_consensus_dag(pcorr_adj: np.ndarray, F_mat: np.ndarray,
                       te: np.ndarray, cols: list, out_dir: Path,
                       granger_thresh: float = 2.0,
                       te_thresh: float = 0.01):
    """
    Consensus DAG: edge exists if ≥ 2 of 3 methods agree.
    Direction from Granger / transfer entropy (asymmetric).
    """
    d = len(cols)

    # Method 1: partial corr (undirected → both directions)
    m1 = (pcorr_adj > 0).astype(float)

    # Method 2: Granger (directed)
    m2 = (F_mat > granger_thresh).astype(float)

    # Method 3: transfer entropy (directed)
    m3 = (te > te_thresh).astype(float)

    consensus = m1 + m2 + m3  # 0 to 3

    # Build directed adjacency: keep edge if consensus >= 2
    adj = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j and consensus[i, j] >= 2:
                adj[i, j] = consensus[i, j] / 3.0

    short_labels = [SHORT[c] for c in cols]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    mask = np.eye(d, dtype=bool)
    sns.heatmap(consensus, ax=ax, cmap="YlOrRd", vmin=0, vmax=3,
                annot=True, fmt=".0f",
                xticklabels=short_labels, yticklabels=short_labels,
                mask=mask, square=True,
                cbar_kws={"label": "# methods agreeing (out of 3)"})
    ax.set_title("Consensus Causal Graph\n(edge if ≥ 2 methods agree, row → col)")
    ax.set_xlabel("Target (effect)")
    ax.set_ylabel("Source (cause)")

    plt.tight_layout()
    path = out_dir / "consensus_dag.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    return adj, consensus


def plot_networkx_dag(adj: np.ndarray, cols: list, out_dir: Path):
    """Draw the consensus DAG as a network graph if networkx is available."""
    try:
        import networkx as nx
    except ImportError:
        print("  [skip] networkx not installed — pip install networkx")
        return

    d = len(cols)
    short_labels = [SHORT[c] for c in cols]

    G = nx.DiGraph()
    for i in range(d):
        G.add_node(i, label=short_labels[i])

    edge_weights = []
    for i in range(d):
        for j in range(d):
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=adj[i, j])
                edge_weights.append(adj[i, j])

    if not edge_weights:
        print("  [skip] No edges in consensus DAG")
        return

    # Layout: inputs on left, targets on right
    pos = {}
    n_inp = len(INPUT_COLS)
    n_tgt = len(TARGET_COLS)
    inp_y = np.linspace(0.9, 0.1, n_inp)
    tgt_y = np.linspace(0.7, 0.3, n_tgt)
    for i in range(n_inp):
        pos[i] = (0.0, inp_y[i])
    for k in range(n_tgt):
        pos[n_inp + k] = (1.0, tgt_y[k])

    # Node colours
    node_colors = ["#4A90D9"] * n_inp + ["#E8A838"] * n_tgt
    node_sizes = [1500] * n_inp + [1500] * n_tgt

    # Edge widths
    ew = np.array(edge_weights)
    ew_scaled = 1.0 + 4.0 * (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 6))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=ew_scaled,
        edge_color=list(ew),
        edge_cmap=plt.cm.Oranges,
        edge_vmin=0, edge_vmax=1,
        arrows=True, arrowsize=25,
        connectionstyle="arc3,rad=0.15",
        min_source_margin=25, min_target_margin=25,
    )
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="black", linewidths=1)
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={i: short_labels[i] for i in range(d)},
        font_size=10, font_weight="bold"
    )

    # Legend
    inp_patch = mpatches.Patch(color="#4A90D9", label="Input features")
    tgt_patch = mpatches.Patch(color="#E8A838", label="Target outputs")
    ax.legend(handles=[inp_patch, tgt_patch], loc="lower center",
              ncol=2, fontsize=9)

    ax.set_title(
        "Consensus Causal DAG from Dataset\n"
        "(Partial Corr + Granger + Transfer Entropy, ≥ 2/3 agreement)",
        fontsize=11
    )
    ax.axis("off")

    plt.tight_layout()
    path = out_dir / "causal_dag_network.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ======================================================================
# Ground truth comparison
# ======================================================================
def compare_ground_truth(consensus: np.ndarray, cols: list):
    """Check which ground-truth edges were recovered."""
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPARISON")
    print("=" * 60)

    col_idx = {c: i for i, c in enumerate(cols)}
    found = 0
    total = len(GROUND_TRUTH_EDGES)

    for src, dst in GROUND_TRUTH_EDGES:
        i = col_idx.get(src)
        j = col_idx.get(dst)
        if i is None or j is None:
            print(f"  {SHORT[src]:>8} → {SHORT[dst]:<8}  [skip — column not in analysis]")
            continue

        score = consensus[i, j]
        status = "✅" if score >= 2 else ("⚠️ " if score >= 1 else "❌")
        if score >= 2:
            found += 1
        print(f"  {SHORT[src]:>8} → {SHORT[dst]:<8}  consensus={score:.0f}/3  {status}")

    print(f"\nRecovered: {found}/{total} ground-truth edges (≥ 2/3 agreement)")
    return found, total


# ======================================================================
# Main
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Data-driven causal discovery on merged dataset")
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__),
                   "merged_eblnn_dataset.csv"),
                   help="Path to merged EBLNN dataset CSV")
    p.add_argument("--n_samples", type=int, default=50000,
                   help="Subsample size (0 = use all, default 50k for speed)")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Significance level for partial correlation")
    p.add_argument("--te_bins", type=int, default=10,
                   help="Number of bins for transfer entropy")
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__),
                   "plots_causality"),
                   help="Output directory for plots")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATA-DRIVEN CAUSAL DISCOVERY")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\nLoading {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Total rows: {len(df):,}")

    # Show scenario breakdown
    if "scenario" in df.columns:
        print("\n  Scenario breakdown:")
        for sc, cnt in df["scenario"].value_counts().items():
            print(f"    {sc:30s}  {cnt:>8,} rows")

    # Subsample for speed
    if args.n_samples > 0 and len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=args.seed)
        print(f"\n  Subsampled to {len(df):,} rows (seed={args.seed})")

    # Keep only analysis columns
    df_analysis = df[ALL_COLS].copy()
    print(f"  Columns: {ALL_COLS}")

    # Quick stats
    print("\n  Descriptive statistics:")
    print(df_analysis.describe().round(4).to_string())

    # ── 1. Partial Correlation ─────────────────────────────────────────
    print("\n" + "-" * 60)
    print("1. PARTIAL CORRELATION ANALYSIS")
    print("-" * 60)
    pcorr, pvals, pcorr_adj = partial_corr_dag(df_analysis, ALL_COLS, alpha=args.alpha)

    print("\nPartial Correlation Matrix:")
    short = [SHORT[c] for c in ALL_COLS]
    pcorr_display = pcorr.copy()
    pcorr_display.index = short
    pcorr_display.columns = short
    print(pcorr_display.round(4).to_string())

    n_edges_pc = int((pcorr_adj > 0).sum() / 2)  # symmetric
    print(f"\nSignificant edges (FDR < {args.alpha}): {n_edges_pc}")

    plot_partial_corr(pcorr, pcorr_adj, ALL_COLS, out_dir)

    # ── 2. Granger Causality ──────────────────────────────────────────
    print("\n" + "-" * 60)
    print("2. CROSS-SECTIONAL GRANGER CAUSALITY")
    print("-" * 60)
    F_mat = granger_cross_sectional(df_analysis, ALL_COLS)

    print("\n-log₁₀(p) matrix  (row causes col, higher = stronger):")
    F_df = pd.DataFrame(F_mat, index=short, columns=short)
    print(F_df.round(1).to_string())

    n_edges_g = int((F_mat > 2.0).sum())  # p < 0.01
    print(f"\nSignificant directed edges (-log₁₀p > 2): {n_edges_g}")

    plot_granger(F_mat, ALL_COLS, out_dir)

    # ── 3. Transfer Entropy ───────────────────────────────────────────
    print("\n" + "-" * 60)
    print("3. TRANSFER ENTROPY (NONLINEAR)")
    print("-" * 60)
    te = binned_transfer_entropy(df_analysis, ALL_COLS, n_bins=args.te_bins)

    print("\nTransfer Entropy matrix (bits, row → col):")
    te_df = pd.DataFrame(te, index=short, columns=short)
    print(te_df.round(4).to_string())

    n_edges_te = int((te > 0.01).sum())
    print(f"\nEdges with TE > 0.01 bits: {n_edges_te}")

    plot_transfer_entropy(te, ALL_COLS, out_dir)

    # ── 4. Consensus ─────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("4. CONSENSUS DAG (≥ 2/3 methods agree)")
    print("-" * 60)
    adj, consensus = plot_consensus_dag(pcorr_adj, F_mat, te, ALL_COLS, out_dir)

    n_consensus = int((consensus >= 2).sum())
    np.fill_diagonal(consensus, 0)
    print(f"\nConsensus directed edges: {n_consensus}")

    plot_networkx_dag(adj, ALL_COLS, out_dir)

    # ── 5. Ground truth comparison ────────────────────────────────────
    found, total = compare_ground_truth(consensus, ALL_COLS)

    # ── 6. Save results ──────────────────────────────────────────────
    import json
    results = {
        "n_samples": len(df_analysis),
        "alpha": args.alpha,
        "n_edges_partial_corr": n_edges_pc,
        "n_edges_granger": n_edges_g,
        "n_edges_transfer_entropy": n_edges_te,
        "n_edges_consensus": n_consensus,
        "ground_truth_recovered": found,
        "ground_truth_total": total,
        "recovery_rate": found / total if total > 0 else 0,
    }
    results_path = out_dir / "causal_discovery_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    print("\n" + "=" * 60)
    print(f"STRUCTURAL CAUSALITY: {found}/{total} ground-truth edges recovered")
    if found >= total - 1:
        print("✅ Strong structural causality confirmed in the dataset")
    elif found >= total // 2:
        print("⚠️  Partial structural causality — some expected edges missing")
    else:
        print("❌ Weak structural causality — dataset may lack causal signal")
    print("=" * 60)


if __name__ == "__main__":
    main()
