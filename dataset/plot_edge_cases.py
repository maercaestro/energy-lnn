"""
Visualise edge-case scenarios vs real data.
Generates one multi-panel figure per scenario + a combined overview.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Config ────────────────────────────────────────────────────────────────
EDGE_CSV = "edge_cases_v2.csv"
REAL_CSV = "cleaned_furnace_data.csv"
OUT_DIR  = "plots_edge_cases"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = ["InletT", "InletFlow", "FGFlow", "DraftP",
                "FGPressure", "Bridgewall", "OP_Damper"]
TARGET_COLS  = ["OutletT", "ExcessO2"]

SCENARIO_INFO = {
    "S1_flame_out":          {"title": "S1 — Flame-Out / Rich Combustion",
                              "key_inputs": ["FGFlow", "OP_Damper", "DraftP"],
                              "color": "#e74c3c"},
    "S2_air_leak":           {"title": "S2 — Air Leak / Excessive Draft",
                              "key_inputs": ["OP_Damper", "DraftP", "FGFlow"],
                              "color": "#3498db"},
    "S3_tube_rupture":       {"title": "S3 — Tube Rupture / Overheating",
                              "key_inputs": ["InletFlow", "FGFlow", "Bridgewall"],
                              "color": "#e67e22"},
    "S4_positive_pressure":  {"title": "S4 — Positive Furnace Pressure",
                              "key_inputs": ["DraftP", "OP_Damper", "FGFlow"],
                              "color": "#9b59b6"},
    "S5_fuel_trip":          {"title": "S5 — Fuel Trip / Cold Furnace",
                              "key_inputs": ["FGFlow", "FGPressure", "InletFlow"],
                              "color": "#2ecc71"},
    "S6_fuel_contamination": {"title": "S6 — Fuel Gas Contamination",
                              "key_inputs": ["FGFlow", "FGPressure", "InletFlow"],
                              "color": "#1abc9c"},
}

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
df_edge = pd.read_csv(EDGE_CSV)
df_real = pd.read_csv(REAL_CSV)
print(f"  Real: {len(df_real):,}   Edge: {len(df_edge):,}")


# ── 1. Per-scenario detail plots ─────────────────────────────────────────
def plot_scenario(s_name, info, df_s, df_r):
    """
    6-panel figure:
      Row 1: Histograms of OutletT, ExcessO2 (edge vs real)
      Row 2: Scatter of OutletT vs key_input[0], ExcessO2 vs key_input[1]
      Row 3: Scatter of OutletT vs ExcessO2, histogram of key_input[2]
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(info["title"], fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                           left=0.08, right=0.95, top=0.92, bottom=0.06)
    c = info["color"]
    ki = info["key_inputs"]

    # ── Row 1: Histograms ──
    # OutletT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_r["OutletT"], bins=80, alpha=0.5, color="gray",
             label="Real", density=True)
    ax1.hist(df_s["OutletT"], bins=80, alpha=0.6, color=c,
             label=s_name, density=True)
    ax1.set_xlabel("OutletT (°C)")
    ax1.set_ylabel("Density")
    ax1.set_title("OutletT Distribution")
    ax1.legend(fontsize=8)

    # ExcessO2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_r["ExcessO2"], bins=80, alpha=0.5, color="gray",
             label="Real", density=True)
    ax2.hist(df_s["ExcessO2"], bins=80, alpha=0.6, color=c,
             label=s_name, density=True)
    ax2.set_xlabel("ExcessO2 (%)")
    ax2.set_ylabel("Density")
    ax2.set_title("ExcessO2 Distribution")
    ax2.legend(fontsize=8)

    # ── Row 2: Key scatter plots ──
    # OutletT vs key_input[0]
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(df_r[ki[0]].values[::20], df_r["OutletT"].values[::20],
                s=1, alpha=0.15, color="gray", label="Real")
    ax3.scatter(df_s[ki[0]], df_s["OutletT"],
                s=3, alpha=0.3, color=c, label=s_name)
    ax3.set_xlabel(ki[0])
    ax3.set_ylabel("OutletT (°C)")
    ax3.set_title(f"OutletT vs {ki[0]}")
    ax3.legend(fontsize=8, markerscale=4)

    # ExcessO2 vs key_input[1]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_r[ki[1]].values[::20], df_r["ExcessO2"].values[::20],
                s=1, alpha=0.15, color="gray", label="Real")
    ax4.scatter(df_s[ki[1]], df_s["ExcessO2"],
                s=3, alpha=0.3, color=c, label=s_name)
    ax4.set_xlabel(ki[1])
    ax4.set_ylabel("ExcessO2 (%)")
    ax4.set_title(f"ExcessO2 vs {ki[1]}")
    ax4.legend(fontsize=8, markerscale=4)

    # ── Row 3: T vs O2 scatter + key_input[2] histogram ──
    # OutletT vs ExcessO2
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(df_r["ExcessO2"].values[::20], df_r["OutletT"].values[::20],
                s=1, alpha=0.15, color="gray", label="Real")
    ax5.scatter(df_s["ExcessO2"], df_s["OutletT"],
                s=3, alpha=0.3, color=c, label=s_name)
    ax5.set_xlabel("ExcessO2 (%)")
    ax5.set_ylabel("OutletT (°C)")
    ax5.set_title("OutletT vs ExcessO2")
    ax5.legend(fontsize=8, markerscale=4)

    # key_input[2] histogram
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df_r[ki[2]], bins=80, alpha=0.5, color="gray",
             label="Real", density=True)
    ax6.hist(df_s[ki[2]], bins=80, alpha=0.6, color=c,
             label=s_name, density=True)
    ax6.set_xlabel(ki[2])
    ax6.set_ylabel("Density")
    ax6.set_title(f"{ki[2]} Distribution")
    ax6.legend(fontsize=8)

    # Stats annotation
    stats_text = (
        f"n={len(df_s):,}  |  "
        f"OutletT: {df_s.OutletT.mean():.1f}±{df_s.OutletT.std():.1f}°C  |  "
        f"O2: {df_s.ExcessO2.mean():.1f}±{df_s.ExcessO2.std():.2f}%"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic",
             color="gray")

    fpath = os.path.join(OUT_DIR, f"{s_name}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  ✅ {fpath}")


# ── 2. Combined overview plot ────────────────────────────────────────────
def plot_overview(df_edge, df_real):
    """
    2-panel overview: OutletT vs ExcessO2 for all scenarios + real,
    and a violin/box comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Edge-Case Augmentation — Overview", fontsize=16,
                 fontweight="bold")

    # ── Panel 1: T vs O2 scatter (all scenarios overlaid) ──
    ax = axes[0, 0]
    ax.scatter(df_real["ExcessO2"].values[::50],
               df_real["OutletT"].values[::50],
               s=1, alpha=0.1, color="gray", label="Real")
    for s_name, info in SCENARIO_INFO.items():
        sub = df_edge[df_edge.scenario == s_name]
        ax.scatter(sub["ExcessO2"], sub["OutletT"],
                   s=3, alpha=0.25, color=info["color"],
                   label=info["title"].split("—")[1].strip())
    ax.set_xlabel("ExcessO2 (%)")
    ax.set_ylabel("OutletT (°C)")
    ax.set_title("OutletT vs ExcessO2 — All Scenarios")
    ax.legend(fontsize=7, markerscale=4, loc="upper right")

    # ── Panel 2: OutletT boxplot by scenario ──
    ax = axes[0, 1]
    labels = []
    data_T = [df_real["OutletT"].values[::50]]
    labels.append("Real")
    colors_box = ["gray"]
    for s_name, info in SCENARIO_INFO.items():
        sub = df_edge[df_edge.scenario == s_name]
        data_T.append(sub["OutletT"].values)
        labels.append(s_name.split("_", 1)[0])
        colors_box.append(info["color"])
    bp = ax.boxplot(data_T, labels=labels, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_ylabel("OutletT (°C)")
    ax.set_title("OutletT by Scenario")
    ax.tick_params(axis="x", rotation=30)

    # ── Panel 3: ExcessO2 boxplot by scenario ──
    ax = axes[1, 0]
    data_O2 = [df_real["ExcessO2"].values[::50]]
    labels2 = ["Real"]
    for s_name, info in SCENARIO_INFO.items():
        sub = df_edge[df_edge.scenario == s_name]
        data_O2.append(sub["ExcessO2"].values)
        labels2.append(s_name.split("_", 1)[0])
    bp2 = ax.boxplot(data_O2, labels=labels2, patch_artist=True, showfliers=False)
    for patch, col in zip(bp2["boxes"], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_ylabel("ExcessO2 (%)")
    ax.set_title("ExcessO2 by Scenario")
    ax.tick_params(axis="x", rotation=30)

    # ── Panel 4: Sample counts + confidence breakdown (stacked bar) ──
    ax = axes[1, 1]
    scenarios = list(SCENARIO_INFO.keys())
    short = [s.split("_", 1)[0] for s in scenarios]
    high_counts, med_counts, low_counts = [], [], []
    for s_name in scenarios:
        sub = df_edge[df_edge.scenario == s_name]
        high_counts.append((sub.confidence == "high").sum())
        med_counts.append((sub.confidence == "medium").sum())
        low_counts.append((sub.confidence == "low").sum())

    x = np.arange(len(scenarios))
    w = 0.6
    ax.bar(x, high_counts, w, label="High conf.", color="#2ecc71", alpha=0.8)
    ax.bar(x, med_counts, w, bottom=high_counts, label="Medium conf.",
           color="#f39c12", alpha=0.8)
    bottom2 = [h + m for h, m in zip(high_counts, med_counts)]
    ax.bar(x, low_counts, w, bottom=bottom2, label="Low conf.",
           color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30)
    ax.set_ylabel("Count")
    ax.set_title("Confidence (Mahalanobis) Breakdown")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fpath = os.path.join(OUT_DIR, "overview_all_scenarios.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  ✅ {fpath}")


# ── 3. DeltaT + FGFlow phase-space plot ──────────────────────────────────
def plot_phase_space(df_edge, df_real):
    """FGFlow vs DeltaT coloured by scenario — shows operating envelope expansion."""
    fig, ax = plt.subplots(figsize=(12, 8))

    df_real["DeltaT"] = df_real["OutletT"] - df_real["InletT"]
    ax.scatter(df_real["FGFlow"].values[::50],
               df_real["DeltaT"].values[::50],
               s=1, alpha=0.1, color="gray", label="Real")

    for s_name, info in SCENARIO_INFO.items():
        sub = df_edge[df_edge.scenario == s_name].copy()
        sub["DeltaT"] = sub["OutletT"] - sub["InletT"]
        ax.scatter(sub["FGFlow"], sub["DeltaT"],
                   s=4, alpha=0.3, color=info["color"],
                   label=info["title"].split("—")[1].strip())

    ax.set_xlabel("FGFlow (Nm³/hr)", fontsize=12)
    ax.set_ylabel("ΔT = OutletT − InletT (°C)", fontsize=12)
    ax.set_title("Operating Envelope: FGFlow vs ΔT", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.3)

    fpath = os.path.join(OUT_DIR, "phase_space_FGFlow_DeltaT.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {fpath}")


# ── Main ──────────────────────────────────────────────────────────────────
print("\n📊 Generating per-scenario plots...")
for s_name, info in SCENARIO_INFO.items():
    df_s = df_edge[df_edge.scenario == s_name]
    plot_scenario(s_name, info, df_s, df_real)

print("\n📊 Generating overview plot...")
plot_overview(df_edge, df_real)

print("\n📊 Generating phase-space plot...")
plot_phase_space(df_edge, df_real)

print(f"\n✅ All plots saved to {OUT_DIR}/")
