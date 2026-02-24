"""
DAG Extraction from the Learned EBM Head
==========================================
Checks whether the energy manifold has captured structural causality
over the input features.

Two complementary analyses:

1. First-order Jacobian  ∂E/∂x_i
   How sensitive is the energy to each input feature?
   → Node importance (which features matter to the energy landscape)

2. Second-order Hessian  ∂²E / ∂x_i ∂x_j
   How strongly does changing x_i alter the energy gradient w.r.t. x_j?
   → Pairwise coupling → skeleton of the causal graph

3. Physics-head Jacobian  ∂ŷ_k / ∂x_i
   Temporal causality: which inputs drive which physics outputs?
   → Directed edges from inputs to next-state targets

DAG direction heuristic
-----------------------
For the EBM Hessian the graph is undirected (H_ij = H_ji by Schwarz).
We break symmetry using domain knowledge ordering:
   fuel_flow → AFR → current_temp → inflow_temp → inflow_rate
   → next_temp, next_O2 → energy
Upper triangle = causal direction (i → j means j causally depends on i).

Usage
-----
    cd energy-lnn/eblnn
    python experiments/dag_from_ebm.py              # uses best_model.pth
    python experiments/dag_from_ebm.py \\
        --model_path results/models/best_model.pth \\
        --data_path  data/synthetic_temperature_data.csv \\
        --n_samples  500 \\
        --threshold  0.05 \\
        --out_dir    results/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to TkAgg if you want live window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import create_model
from src.data import DataPipeline, INPUT_COLS, TARGET_COLS

# ── Feature & node labels ──────────────────────────────────────────────────
INPUT_LABELS  = ["fuel_flow", "AFR", "cur_temp", "inflow_T", "inflow_rate"]
TARGET_LABELS = ["next_temp", "next_O2"]
ENERGY_LABEL  = "Energy E"

ALL_NODES = INPUT_LABELS + TARGET_LABELS + [ENERGY_LABEL]

# Causal ordering index (domain knowledge: inputs → targets → energy)
CAUSAL_ORDER = {n: i for i, n in enumerate(ALL_NODES)}


# ──────────────────────────────────────────────────────────────────────────
# Jacobian utilities
# ──────────────────────────────────────────────────────────────────────────

def batch_jacobian(
    fn,
    x: torch.Tensor,
    output_idx: int | None = None,
) -> torch.Tensor:
    """
    Compute the Jacobian ∂fn(x) / ∂x, averaged over the batch.

    Parameters
    ----------
    fn          : function x → (B,) scalar or (B, K) vector
    x           : (B, T, D)  input batch (requires_grad will be set)
    output_idx  : if fn returns (B, K), select output dimension K=output_idx
                  if None, fn must return (B,)

    Returns
    -------
    mean_grad : (D,) averaged absolute gradient per input feature
    """
    x = x.detach().clone().requires_grad_(True)
    out = fn(x)                          # (B,) or (B, K)
    if output_idx is not None:
        out = out[:, output_idx]         # (B,)
    out_sum = out.sum()
    out_sum.backward()
    grad = x.grad.abs()                  # (B, T, D)
    return grad.mean(dim=(0, 1))         # (D,)


def batch_hessian_diag_cross(
    energy_fn,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Hessian ∂²E / ∂x_i ∂x_j averaged over the batch.

    Returns the (D, D) mean absolute Hessian matrix.
    Only the cross-feature terms are meaningful for the DAG.

    Note: Computing the full Hessian is O(D²) backward passes.
    For D=5 this is fast.
    """
    D = x.shape[-1]
    H = torch.zeros(D, D)

    for i in range(D):
        # First derivative w.r.t. x_i
        xi = x.detach().clone().requires_grad_(True)
        E = energy_fn(xi)               # (B,)
        grad_i = torch.autograd.grad(
            E.sum(), xi, create_graph=True
        )[0]                            # (B, T, D)
        g_i = grad_i[..., i]            # (B, T)

        for j in range(D):
            # Second derivative w.r.t. x_j
            xj = x.detach().clone().requires_grad_(True)
            Ej = energy_fn(xj)
            grad_ij = torch.autograd.grad(
                Ej.sum(), xj, create_graph=True
            )[0]
            g_ij = grad_ij[..., j]

            # Cross term: how does ∂E/∂x_j change as x_i changes?
            # Approximate via finite difference on the gradient
            eps = 1e-3
            x_pert = x.detach().clone()
            x_pert[..., i] += eps
            x_pert.requires_grad_(True)
            E_pert = energy_fn(x_pert)
            grad_pert = torch.autograd.grad(E_pert.sum(), x_pert)[0]
            g_pert_j = grad_pert[..., j]

            x_base = x.detach().clone().requires_grad_(True)
            E_base = energy_fn(x_base)
            grad_base = torch.autograd.grad(E_base.sum(), x_base)[0]
            g_base_j = grad_base[..., j]

            hess_ij = ((g_pert_j - g_base_j) / eps).abs().mean().item()
            H[i, j] = hess_ij

    return H


def physics_jacobian(model, x: torch.Tensor) -> torch.Tensor:
    """
    Compute |∂ŷ_k / ∂x_i| averaged over batch and time.

    Returns (K, D) — K outputs, D inputs.
    """
    x = x.detach().clone().requires_grad_(True)
    h_seq, _ = model.cfc_body(x)
    phys = model.phys_head(h_seq)    # (B, T, K)

    K = phys.shape[-1]
    D = x.shape[-1]
    J = torch.zeros(K, D)

    for k in range(K):
        if x.grad is not None:
            x.grad.zero_()
        phys[..., k].sum().backward(retain_graph=(k < K - 1))
        J[k] = x.grad.abs().mean(dim=(0, 1))
        x.grad.zero_()

    return J.detach()


# ──────────────────────────────────────────────────────────────────────────
# Build adjacency matrix
# ──────────────────────────────────────────────────────────────────────────

def build_adjacency(
    ebm_jacobian: np.ndarray,    # (D,)
    hessian: np.ndarray,         # (D, D)
    phys_jacobian: np.ndarray,   # (K, D)
    threshold: float,
) -> np.ndarray:
    """
    Construct a directed adjacency matrix over ALL_NODES.

    Node layout:
        0..D-1     input features
        D..D+K-1   physics targets
        D+K        energy node

    Edge types
    ----------
    input_i → energy      : |∂E/∂x_i| > threshold      (first-order Jacobian)
    input_i ↔ input_j     : H[i,j] > threshold           (Hessian coupling)
                            direction: lower causal-order → higher
    input_i → target_k    : |∂ŷ_k/∂x_i| > threshold    (physics Jacobian)
    target_k → energy     : always add (targets feed the energy head)
    """
    D = len(ebm_jacobian)
    K = phys_jacobian.shape[0]
    N = D + K + 1
    A = np.zeros((N, N))

    energy_idx = N - 1
    target_idxs = list(range(D, D + K))

    # 1. Input → Energy  (first-order Jacobian)
    for i in range(D):
        if ebm_jacobian[i] > threshold:
            A[i, energy_idx] = ebm_jacobian[i]

    # 2. Input ↔ Input coupling  (Hessian, upper triangle = i→j)
    h_max = hessian.max() if hessian.max() > 0 else 1.0
    for i in range(D):
        for j in range(i + 1, D):
            val = (hessian[i, j] + hessian[j, i]) / 2.0
            if val / h_max > threshold:
                # Direct edge from lower → higher causal order
                A[i, j] = val / h_max

    # 3. Input → Physics target  (physics Jacobian)
    pj_max = phys_jacobian.max() if phys_jacobian.max() > 0 else 1.0
    for k in range(K):
        for i in range(D):
            val = phys_jacobian[k, i] / pj_max
            if val > threshold:
                A[i, target_idxs[k]] = val

    # 4. Physics targets → Energy  (structural: targets are part of energy input)
    for t_idx in target_idxs:
        A[t_idx, energy_idx] = 1.0

    return A


# ──────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────

def plot_dag(
    A: np.ndarray,
    node_labels: list[str],
    ebm_jacobian: np.ndarray,
    out_path: Path,
) -> None:
    """Draw the DAG with edge weights and node importance colouring."""
    try:
        import networkx as nx
    except ImportError:
        print("[warn] networkx not installed — skipping DAG plot. pip install networkx")
        return

    D = len(ebm_jacobian)
    N = len(node_labels)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    edge_weights = []
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])
                edge_weights.append(A[i, j])

    # Layout: inputs left, targets middle, energy right
    pos = {}
    input_y  = np.linspace(0.9, 0.1, D)
    target_y = np.linspace(0.75, 0.25, N - D - 1)
    for i in range(D):
        pos[i] = (0.0, input_y[i])
    for k, t_idx in enumerate(range(D, N - 1)):
        pos[t_idx] = (0.5, target_y[k])
    pos[N - 1] = (1.0, 0.5)   # energy node

    # Node colours: inputs by |∂E/∂x_i|, targets gold, energy red
    jac_norm = ebm_jacobian / (ebm_jacobian.max() + 1e-9)
    cmap = plt.cm.Blues
    node_colors = []
    for i in range(N):
        if i < D:
            node_colors.append(cmap(0.3 + 0.7 * jac_norm[i]))
        elif i < N - 1:
            node_colors.append("gold")
        else:
            node_colors.append("tomato")

    node_sizes = [1200] * D + [1000] * (N - D - 1) + [1800]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Edges
    if edge_weights:
        ew_arr = np.array(edge_weights)
        ew_norm = 1.0 + 4.0 * (ew_arr - ew_arr.min()) / (ew_arr.max() - ew_arr.min() + 1e-9)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=ew_norm,
            edge_color=list(ew_arr),
            edge_cmap=plt.cm.Oranges,
            edge_vmin=0, edge_vmax=1,
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=20,
            min_target_margin=20,
        )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={i: node_labels[i] for i in range(N)},
        font_size=9, font_weight="bold"
    )

    ax.set_title(
        "DAG from Learned EBM Head\n"
        "(node colour = |∂E/∂x| importance, edge width = coupling strength)",
        fontsize=11
    )
    ax.axis("off")

    # Colourbar for node energy sensitivity
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="|∂E/∂x| normalised")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"DAG saved → {out_path}")


def plot_hessian(H: np.ndarray, labels: list[str], out_path: Path) -> None:
    """Heatmap of the pairwise energy interaction (Hessian) matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H, cmap="RdBu_r", vmin=-H.max(), vmax=H.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{H[i,j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="|∂²E / ∂x_i ∂x_j|")
    ax.set_title("EBM Hessian — Pairwise Input Coupling\n(off-diagonal = interaction strength)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Hessian saved → {out_path}")


def plot_jacobian_bar(
    ebm_jac: np.ndarray,
    phys_jac: np.ndarray,
    input_labels: list[str],
    target_labels: list[str],
    out_path: Path,
) -> None:
    """Bar chart: first-order Jacobians for energy and physics heads."""
    fig, axes = plt.subplots(1, len(target_labels) + 1, figsize=(4 * (len(target_labels) + 1), 4))

    # EBM head
    ax = axes[0]
    norm = ebm_jac / (ebm_jac.max() + 1e-9)
    bars = ax.barh(input_labels, norm, color="tomato")
    ax.set_title("∂E / ∂x\n(energy sensitivity)")
    ax.set_xlabel("Normalised |gradient|")
    ax.axvline(0.05, color="grey", linestyle="--", linewidth=0.8, label="threshold")
    ax.legend(fontsize=7)

    # Physics head
    for k, t_label in enumerate(target_labels):
        ax = axes[k + 1]
        j_norm = phys_jac[k] / (phys_jac[k].max() + 1e-9)
        ax.barh(input_labels, j_norm, color="steelblue")
        ax.set_title(f"∂{t_label} / ∂x\n(physics sensitivity)")
        ax.set_xlabel("Normalised |gradient|")
        ax.axvline(0.05, color="grey", linestyle="--", linewidth=0.8)

    plt.suptitle("First-order Jacobians — Energy & Physics Heads", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Jacobian bars saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAG extraction from trained EBM head")
    p.add_argument("--model_path", default=str(ROOT / "results/models/best_model.pth"))
    p.add_argument("--data_path",  default=str(ROOT / "data/synthetic_temperature_data.csv"))
    p.add_argument("--n_samples",  type=int,   default=500)
    p.add_argument("--threshold",  type=float, default=0.05,
                   help="Min normalised coupling to draw an edge")
    p.add_argument("--out_dir",    default=str(ROOT / "results/plots"))
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--ebm_hidden_dims", default="[128,64]")
    return p.parse_args()


def main() -> None:
    import ast
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ───────────────────────────────────────────────────
    print(f"Loading model from {args.model_path}")
    ebm_dims = ast.literal_eval(args.ebm_hidden_dims)
    model = create_model(
        input_size=5,
        hidden_size=args.hidden_size,
        phys_output_size=2,
        ebm_hidden_dims=ebm_dims,
        device="cpu",
    )
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── 2. Load data sample ─────────────────────────────────────────────
    print(f"Loading data from {args.data_path}")
    pipeline = DataPipeline(
        data_path=args.data_path,
        num_sequences=max(1000, args.n_samples * 2),
        force_regenerate=False,
    ).build()

    # Pull a sample from the test loader
    x_samples, y_samples = [], []
    for xb, yb in pipeline.test_loader:
        x_samples.append(xb)
        y_samples.append(yb)
        if sum(len(x) for x in x_samples) >= args.n_samples:
            break

    x = torch.cat(x_samples, dim=0)[: args.n_samples]   # (N, T, 5)
    y = torch.cat(y_samples, dim=0)[: args.n_samples]

    print(f"Using {x.shape[0]} sequences  (shape {tuple(x.shape)})")

    # ── 3. First-order Jacobian ∂E/∂x ──────────────────────────────────
    print("Computing EBM first-order Jacobian ∂E/∂x …")
    ebm_jac = batch_jacobian(model.energy, x).numpy()   # (D,)

    # ── 4. Physics head Jacobian ∂ŷ/∂x ─────────────────────────────────
    print("Computing physics head Jacobian ∂ŷ/∂x …")
    phys_jac = physics_jacobian(model, x).numpy()       # (K, D)

    # ── 5. Hessian ∂²E/∂x_i∂x_j ─────────────────────────────────────────
    # Use a smaller subset (Hessian is expensive)
    print("Computing EBM Hessian (this may take ~30 s) …")
    x_small = x[:min(100, args.n_samples)]
    H = batch_hessian_diag_cross(model.energy, x_small).numpy()   # (D, D)

    # ── 6. Print summary ─────────────────────────────────────────────────
    print("\n=== First-order Jacobian  ∂E/∂x  (normalised) ===")
    ebm_norm = ebm_jac / (ebm_jac.max() + 1e-9)
    for name, val in zip(INPUT_LABELS, ebm_norm):
        bar = "█" * int(val * 30)
        flag = " ← above threshold" if val > args.threshold else ""
        print(f"  {name:<14} {val:.4f}  {bar}{flag}")

    print("\n=== Physics Jacobians  ∂ŷ/∂x  (normalised) ===")
    for k, t in enumerate(TARGET_LABELS):
        pj_norm = phys_jac[k] / (phys_jac[k].max() + 1e-9)
        print(f"\n  Target: {t}")
        for name, val in zip(INPUT_LABELS, pj_norm):
            bar = "█" * int(val * 30)
            print(f"    {name:<14} {val:.4f}  {bar}")

    print("\n=== Hessian  (pairwise EBM coupling) ===")
    h_max = H.max()
    for i, ni in enumerate(INPUT_LABELS):
        for j, nj in enumerate(INPUT_LABELS):
            if i < j and H[i, j] / (h_max + 1e-9) > args.threshold:
                print(f"  {ni}  ↔  {nj}   H={H[i,j]:.4f}")

    # ── 7. Build adjacency matrix ─────────────────────────────────────────
    print("\nBuilding adjacency matrix …")
    A = build_adjacency(ebm_jac, H, phys_jac, threshold=args.threshold)

    # Count inferred edges
    n_edges = (A > 0).sum()
    print(f"Edges found: {n_edges}")

    edge_flag = "✓ Structural causality detected" if n_edges > len(INPUT_LABELS) else \
                "⚠  Sparse graph — EBM may not have learned rich causal structure yet"
    print(edge_flag)

    # ── 8. Plots ──────────────────────────────────────────────────────────
    plot_jacobian_bar(ebm_jac, phys_jac, INPUT_LABELS, TARGET_LABELS,
                      out_dir / "jacobian_bars.png")

    plot_hessian(H, INPUT_LABELS,
                 out_dir / "ebm_hessian.png")

    plot_dag(A, ALL_NODES, ebm_jac,
             out_dir / "causal_dag.png")

    # ── 9. Save raw arrays for further analysis ───────────────────────────
    np.save(out_dir / "ebm_jacobian.npy",  ebm_jac)
    np.save(out_dir / "ebm_hessian.npy",   H)
    np.save(out_dir / "phys_jacobian.npy", phys_jac)
    np.save(out_dir / "adjacency.npy",     A)
    print(f"\nArrays saved to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
