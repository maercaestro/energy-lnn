"""
run_landscape.py
================
Entry point for the H1 energy landscape analysis.

Loads a trained EB-LNN checkpoint, computes the learned energy for normal
and edge-case furnace sequences, and produces six publication-quality
figures that constitute the empirical evidence for H1:

  H1: The EB-LNN energy head embeds thermodynamic structural constraints,
      assigning low energy to physically safe states and high energy to
      physically dangerous (edge-case) states.

Usage
-----
  # defaults (local run, seed 42 checkpoint)
  python eblnn-landscape/scripts/run_landscape.py

  # VM run — override paths and disable W&B interactive prompt
  python eblnn-landscape/scripts/run_landscape.py \\
      --config eblnn-landscape/config/landscape_config.yaml \\
      --checkpoint eblnn-benchmark/results/default/eblnn/seed_42/models/best_model.pth \\
      --real_csv   dataset/real_furnace_eblnn.csv \\
      --edge_csv   dataset/edge_cases_v2_eblnn.csv \\
      --out_dir    eblnn-landscape/results \\
      --run_name   landscape_seed42 \\
      --device     cpu \\
      --max_seqs   2000 \\
      --no_wandb

  # Full run (all sequences, online W&B)
  python eblnn-landscape/scripts/run_landscape.py \\
      --max_seqs null

Output
------
  results/
    fig1_energy_violin.png
    fig2_energy_cdf.png
    fig3_2d_contours.png
    fig4_1d_profiles.png
    fig5_energy_vs_safety.png
    fig6_separation_margins.png
    energy_arrays.npz          (raw energies per scenario)
    summary.json               (stats + sensitivity ranking)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — safe for VM / tmux

import numpy as np
import yaml

# ── project root + landscape src on sys.path ──────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
LANDSCAPE_DIR = ROOT / "eblnn-landscape"
for p in (str(ROOT), str(LANDSCAPE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import (  # noqa: E402
    load_model,
    fit_scaler,
    build_scenario_sequences,
    build_grid_probe,
    compute_energies_per_scenario,
    compute_energy_grid,
    compute_all_1d_profiles,
    compute_sensitivity_ranking,
    compute_energy_stats,
    compute_separation_margin,
    plot_energy_distributions,
    plot_energy_cdf,
    plot_2d_contours,
    plot_1d_profiles,
    plot_energy_vs_safety,
    plot_separation_margins,
    SCENARIO_ORDER,
    SCENARIO_LABELS,
)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EB-LNN H1 energy landscape analysis"
    )
    p.add_argument("--config",      default="eblnn-landscape/config/landscape_config.yaml")
    p.add_argument("--checkpoint",  default=None, help="Override checkpoint path")
    p.add_argument("--real_csv",    default=None)
    p.add_argument("--edge_csv",    default=None)
    p.add_argument("--out_dir",     default=None)
    p.add_argument("--run_name",    default=None)
    p.add_argument("--device",      default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--max_seqs",    default=None,
                   help="Max sequences per group (int or 'null' for all)")
    p.add_argument("--no_wandb",    action="store_true", help="Disable W&B logging")
    p.add_argument("--grid_res",    type=int, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        import torch
        return "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return device_str


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    args = parse_args()

    # ── load config ───────────────────────────────────────────────────────
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)

    # CLI overrides
    run_name = args.run_name or cfg.get("run_name", "landscape_default")
    checkpoint_path = args.checkpoint or (ROOT / cfg["model"]["checkpoint_path"])
    real_csv  = args.real_csv  or (ROOT / cfg["data"]["real_csv"])
    edge_csv  = args.edge_csv  or (ROOT / cfg["data"]["edge_csv"])
    out_dir   = Path(args.out_dir or (ROOT / cfg["plots"]["out_dir"])) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(
        args.device or cfg["analysis"].get("device", "auto")
    )

    max_seqs_str = args.max_seqs
    if max_seqs_str is None:
        max_seqs = cfg["data"].get("max_sequences_per_group", 2000)
    elif max_seqs_str.lower() == "null":
        max_seqs = None
    else:
        max_seqs = int(max_seqs_str)

    grid_res  = args.grid_res or cfg["analysis"].get("grid_res", 60)
    n_sigma   = cfg["analysis"].get("n_sigma", 2.5)
    batch_sz  = cfg["analysis"].get("batch_size", 512)
    seq_len   = cfg["data"].get("seq_len", 30)
    stride    = cfg["data"].get("stride", 30)
    seed      = cfg["data"].get("seed", 42)
    dpi       = cfg["plots"].get("dpi", 150)
    fmt       = cfg["plots"].get("format", "png")

    use_wandb = (not args.no_wandb) and cfg.get("wandb", {}).get("enabled", True)

    print(f"\n{'='*60}")
    print(f"  EB-LNN Energy Landscape Analysis — H1")
    print(f"{'='*60}")
    print(f"  run_name   : {run_name}")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  device     : {device}")
    print(f"  max_seqs   : {max_seqs}")
    print(f"  out_dir    : {out_dir}")
    print(f"  W&B        : {use_wandb}")
    print(f"{'='*60}\n")

    # ── W&B init ──────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wcfg = cfg.get("wandb", {})
        wandb.init(
            project=wcfg.get("project", "eblnn-landscape"),
            entity=wcfg.get("entity"),
            name=run_name,
            mode=wcfg.get("mode", "online"),
            tags=wcfg.get("tags", []),
            config={
                "checkpoint": str(checkpoint_path),
                "device": device,
                "max_seqs": max_seqs,
                "grid_res": grid_res,
                "seq_len": seq_len,
            },
        )

    # ── load model ────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = load_model(
        checkpoint_path=checkpoint_path,
        hidden_size=model_cfg.get("hidden_size", 128),
        ebm_hidden_dims=model_cfg.get("ebm_hidden_dims", [128, 64]),
        input_size=model_cfg.get("input_size", 5),
        phys_output_size=model_cfg.get("phys_output_size", 2),
        mixed_memory=model_cfg.get("mixed_memory", True),
        device=device,
    )

    # ── fit scaler ────────────────────────────────────────────────────────
    scaler = fit_scaler(
        real_csv=real_csv,
        train_fraction=cfg["data"].get("scaler_train_fraction", 0.8),
        seed=seed,
    )

    # ── build sequences ───────────────────────────────────────────────────
    scenario_sequences = build_scenario_sequences(
        real_csv=real_csv,
        edge_csv=edge_csv,
        scaler=scaler,
        seq_len=seq_len,
        stride=stride,
        max_per_group=max_seqs,
        seed=seed,
    )

    # ── compute energies ──────────────────────────────────────────────────
    print("\n[main] Computing energies ...")
    scenario_energies = compute_energies_per_scenario(
        model=model,
        scenario_sequences=scenario_sequences,
        device=device,
        batch_size=batch_sz,
    )

    # ── stats ─────────────────────────────────────────────────────────────
    stats = compute_energy_stats(scenario_energies)
    margins = compute_separation_margin(scenario_energies, reference="real")

    print("\n[main] Energy separation margins (vs Normal):")
    for s, m in margins.items():
        sign = "↑ HIGHER" if m > 0 else "↓ lower"
        print(f"  {SCENARIO_LABELS.get(s,s):<22s}  ΔE = {m:+.4f}  {sign}")

    # ── 1-D profiles ──────────────────────────────────────────────────────
    input_labels = ["fuel_flow", "AFR", "cur_temp", "inflow_T", "inflow_rate"]
    x_all = np.concatenate(
        [scenario_sequences["real"][:, 0, :] for _ in [1]]  # (N, D)
    )
    x_mean = x_all.mean(axis=0)   # shape (D,)
    x_std  = x_all.std(axis=0).clip(1e-6)

    print("\n[main] Computing 1-D energy profiles ...")
    profiles = compute_all_1d_profiles(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        seq_len=seq_len,
        input_labels=input_labels,
        n_sigma=n_sigma,
        n_points=100,
        device=device,
    )
    sensitivity = compute_sensitivity_ranking(profiles)
    print("\n[main] Sensitivity ranking:")
    for rank, (name, val) in enumerate(sensitivity.items(), 1):
        print(f"  {rank}. {name:<20s}  |∂E/∂x| = {val:.4f}")

    # ── 2-D contour grids ─────────────────────────────────────────────────
    # Three most physically meaningful pairs for a furnace:
    #   fuel_flow (0) vs cur_temp (2)  — combustion intensity
    #   AFR (1)       vs fuel_flow (0) — air/fuel balance
    #   cur_temp (2)  vs inflow_T (3)  — heat transfer
    contour_pairs = [(0, 2), (1, 0), (2, 3)]
    grids: dict = {}
    print("\n[main] Computing 2-D energy grids ...")
    for di, dj in contour_pairs:
        xi_v, xj_v, probes = build_grid_probe(
            x_mean, x_std, di, dj, grid_res, seq_len, n_sigma
        )
        E_grid = compute_energy_grid(model, probes, grid_res, device, batch_sz)
        grids[(di, dj)] = (xi_v, xj_v, E_grid)
        print(f"  ({input_labels[di]}, {input_labels[dj]}) grid done  "
              f"min={E_grid.min():.2f}  max={E_grid.max():.2f}")

    # ── generate figures ──────────────────────────────────────────────────
    print("\n[main] Generating figures ...")
    figures = {}

    figures["fig1_energy_violin"] = plot_energy_distributions(scenario_energies)
    figures["fig2_energy_cdf"]    = plot_energy_cdf(scenario_energies)
    figures["fig3_2d_contours"]   = plot_2d_contours(
        grids, scenario_sequences, input_labels=input_labels
    )
    figures["fig4_1d_profiles"]   = plot_1d_profiles(profiles, sensitivity)
    figures["fig5_energy_vs_safety"] = plot_energy_vs_safety(
        scenario_energies, scenario_sequences, target_scaler=None
    )
    figures["fig6_separation_margins"] = plot_separation_margins(margins)

    # ── save figures ──────────────────────────────────────────────────────
    for name, fig in figures.items():
        fpath = out_dir / f"{name}.{fmt}"
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"  saved: {fpath.relative_to(ROOT)}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── save raw energies ─────────────────────────────────────────────────
    if cfg.get("output", {}).get("save_energies_npz", True):
        npz_path = out_dir / "energy_arrays.npz"
        np.savez(npz_path, **scenario_energies)
        print(f"  saved: {npz_path.relative_to(ROOT)}")

    # ── save summary JSON ─────────────────────────────────────────────────
    summary = {
        "run_name":   run_name,
        "checkpoint": str(checkpoint_path),
        "device":     device,
        "stats":      stats,
        "margins":    margins,
        "sensitivity_ranking": sensitivity,
        "h1_evidence": {
            "all_margins_positive": all(v > 0 for v in margins.values()),
            "max_margin_scenario":  max(margins, key=margins.get),
            "max_margin_value":     max(margins.values()),
            "real_mean_energy":     float(scenario_energies["real"].mean()),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {summary_path.relative_to(ROOT)}")

    # ── W&B log ───────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        import matplotlib.pyplot as plt

        # log figures
        for name, fpath in [(n, out_dir / f"{n}.{fmt}") for n in figures]:
            wandb.log({name: wandb.Image(str(fpath))})

        # log scalar metrics
        for s, m in margins.items():
            wandb.log({f"margin/{SCENARIO_LABELS.get(s,s).replace(' ','_')}": m})
        for name, val in sensitivity.items():
            wandb.log({f"sensitivity/{name}": val})
        wandb.log({"h1/all_margins_positive":
                   int(summary["h1_evidence"]["all_margins_positive"])})
        wandb.log({"h1/max_margin": summary["h1_evidence"]["max_margin_value"]})

        # log energy stats as a table
        table = wandb.Table(
            columns=["scenario", "mean", "std", "median", "n"]
        )
        for s in SCENARIO_ORDER:
            if s in stats:
                st = stats[s]
                table.add_data(
                    SCENARIO_LABELS.get(s, s),
                    st["mean"], st["std"], st["median"], st["n"]
                )
        wandb.log({"energy_stats_table": table})
        wandb.finish()

    # ── final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  H1 Evidence Summary")
    print(f"{'='*60}")
    h1 = summary["h1_evidence"]
    if h1["all_margins_positive"]:
        print(f"  ✓ ALL {len(margins)} edge-case scenarios have HIGHER energy than normal")
    else:
        neg = [s for s, v in margins.items() if v <= 0]
        print(f"  ✗ {len(neg)} scenario(s) did NOT have higher energy: {neg}")
    print(f"  Normal mean energy : {h1['real_mean_energy']:+.4f}")
    print(f"  Max margin         : {h1['max_margin_value']:+.4f}  ({h1['max_margin_scenario']})")
    print(f"  Most sensitive feat: {list(sensitivity.keys())[0]}")
    print(f"  Completed in       : {elapsed:.1f}s")
    print(f"  Results saved to   : {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
