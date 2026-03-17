"""
Analyze real furnace data for naturally-occurring edge-case conditions.
Applies the same scenario definitions used in generate_edge_cases.py
to the cleaned_furnace_data.csv to quantify how many (if any) rows
in the original plant data resemble each failure mode.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_furnace_data.csv")
n = len(df)
print(f"Total rows: {n:,}\n")

# ── Real Data Statistics ──────────────────────────────────────────────
cols = ["InletT", "InletFlow", "FGFlow", "DraftP", "FGPressure",
        "Bridgewall", "OP_Damper", "OutletT", "ExcessO2"]

print("=== Real Data Statistics ===")
print(f"{'Column':>14s} {'min':>10s} {'p01':>10s} {'mean':>10s} "
      f"{'p99':>10s} {'max':>10s} {'std':>10s}")
for c in cols:
    print(f"{c:>14s} {df[c].min():10.2f} {df[c].quantile(0.01):10.2f} "
          f"{df[c].mean():10.2f} {df[c].quantile(0.99):10.2f} "
          f"{df[c].max():10.2f} {df[c].std():10.2f}")

print()
print("=" * 80)
print("=== Edge-Case Detection in Real Data ===")
print("=" * 80)

# ── S1: Flame-Out / Rich Combustion ──────────────────────────────────
# Edge-case generator bounds: FGFlow [1800, 2500], Damper [36, 42], Draft [-2, 0.5]
s1_strict = (df["FGFlow"] > 1800) | (
    (df["OP_Damper"] < 42) & (df["DraftP"] > -2.0)
)
s1_soft = (df["FGFlow"] > 1600) & (df["OP_Damper"] < 45) & (df["DraftP"] > -2.5)

print(f"\nS1 Flame-Out (rich combustion)")
print(f"   Edge-case bounds: FGFlow>1800 OR (Damper<42 AND Draft>-2)")
print(f"   Strict:  {s1_strict.sum():>8,} rows  ({100*s1_strict.sum()/n:.3f}%)")
print(f"   Soft:    {s1_soft.sum():>8,} rows  ({100*s1_soft.sum()/n:.3f}%)")

# ── S2: Air Leak / Excessive Draft ───────────────────────────────────
# Bounds: Damper [63, 68], Draft [-12, -9], FGFlow [800, 1200]
s2_strict = (df["OP_Damper"] > 63) & (df["DraftP"] < -9.0) & (df["FGFlow"] < 1200)
s2_soft = (df["OP_Damper"] > 60) & (df["DraftP"] < -7.0)

print(f"\nS2 Air Leak (excessive draft)")
print(f"   Edge-case bounds: Damper>63 AND Draft<-9 AND FGFlow<1200")
print(f"   Strict:  {s2_strict.sum():>8,} rows  ({100*s2_strict.sum()/n:.3f}%)")
print(f"   Soft:    {s2_soft.sum():>8,} rows  ({100*s2_soft.sum()/n:.3f}%)")

# ── S3: Tube Rupture / Overheating ───────────────────────────────────
# Bounds: InletFlow [245, 310], FGFlow [1600, 2000], Bridgewall [700, 800]
s3_strict = (df["InletFlow"] < 310) & (df["FGFlow"] > 1600) & (df["Bridgewall"] > 700)
s3_soft = (df["Bridgewall"] > 680) & (df["FGFlow"] > 1500)

print(f"\nS3 Tube Rupture (overheating)")
print(f"   Edge-case bounds: Flow<310 AND FGFlow>1600 AND Bridgewall>700")
print(f"   Strict:  {s3_strict.sum():>8,} rows  ({100*s3_strict.sum()/n:.3f}%)")
print(f"   Soft:    {s3_soft.sum():>8,} rows  ({100*s3_soft.sum()/n:.3f}%)")

# ── S4: Positive Furnace Pressure (blowback) ─────────────────────────
# Bounds: Draft [-0.5, 3.0], Damper [36, 45]
s4_strict = df["DraftP"] > -0.5
s4_soft = (df["DraftP"] > -1.5) & (df["OP_Damper"] < 45)

print(f"\nS4 Positive Pressure (blowback)")
print(f"   Edge-case bounds: Draft > -0.5")
print(f"   Strict:  {s4_strict.sum():>8,} rows  ({100*s4_strict.sum()/n:.3f}%)")
print(f"   Soft:    {s4_soft.sum():>8,} rows  ({100*s4_soft.sum()/n:.3f}%)")

# ── S5: Fuel Trip / Cold Furnace ─────────────────────────────────────
# Bounds: FGFlow [0, 500], FGPressure [0.2, 0.8]
s5_strict = df["FGFlow"] < 500
s5_soft = (df["FGFlow"] < 800) & (df["FGPressure"] < 0.8)

print(f"\nS5 Fuel Trip (loss of firing)")
print(f"   Edge-case bounds: FGFlow<500")
print(f"   Strict:  {s5_strict.sum():>8,} rows  ({100*s5_strict.sum()/n:.3f}%)")
print(f"   Soft:    {s5_soft.sum():>8,} rows  ({100*s5_soft.sum()/n:.3f}%)")

# ── S6: Fuel Gas Contamination ───────────────────────────────────────
# Bounds: FGFlow [800, 1200], FGPressure [0.8, 1.3]
s6_strict = df["FGFlow"].between(800, 1200) & (df["FGPressure"] < 1.3)
s6_soft = (df["FGPressure"] < 1.1) & (df["FGFlow"] < 1200)

print(f"\nS6 Fuel Contamination (low effective LHV)")
print(f"   Edge-case bounds: FGFlow in [800,1200] AND FGPress<1.3")
print(f"   Strict:  {s6_strict.sum():>8,} rows  ({100*s6_strict.sum()/n:.3f}%)")
print(f"   Soft:    {s6_soft.sum():>8,} rows  ({100*s6_soft.sum()/n:.3f}%)")

# ── Combined ─────────────────────────────────────────────────────────
any_strict = s1_strict | s2_strict | s3_strict | s4_strict | s5_strict | s6_strict
any_soft = s1_soft | s2_soft | s3_soft | s4_soft | s5_soft | s6_soft

print(f"\n{'=' * 80}")
print(f"SUMMARY (strict thresholds)")
print(f"{'=' * 80}")
print(f"  S1 Flame-Out:      {s1_strict.sum():>8,}  ({100*s1_strict.sum()/n:.3f}%)")
print(f"  S2 Air Leak:       {s2_strict.sum():>8,}  ({100*s2_strict.sum()/n:.3f}%)")
print(f"  S3 Tube Rupture:   {s3_strict.sum():>8,}  ({100*s3_strict.sum()/n:.3f}%)")
print(f"  S4 Pos. Pressure:  {s4_strict.sum():>8,}  ({100*s4_strict.sum()/n:.3f}%)")
print(f"  S5 Fuel Trip:      {s5_strict.sum():>8,}  ({100*s5_strict.sum()/n:.3f}%)")
print(f"  S6 Contamination:  {s6_strict.sum():>8,}  ({100*s6_strict.sum()/n:.3f}%)")
print(f"  ─────────────────────────────────")
print(f"  ANY edge case:     {any_strict.sum():>8,}  ({100*any_strict.sum()/n:.2f}%)")
print(f"  Normal operation:  {(~any_strict).sum():>8,}  ({100*(~any_strict).sum()/n:.2f}%)")

# ── Output-side anomalies ─────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("=== Output-Side Anomalies ===")
print(f"{'=' * 80}")

print("\nExcess O2 extremes:")
for thr in [1.0, 2.0]:
    cnt = (df["ExcessO2"] < thr).sum()
    print(f"  ExcessO2 < {thr:.0f}%:   {cnt:>8,} rows  ({100*cnt/n:.3f}%)")
for thr in [6.0, 8.0, 10.0]:
    cnt = (df["ExcessO2"] > thr).sum()
    print(f"  ExcessO2 > {thr:.0f}%:   {cnt:>8,} rows  ({100*cnt/n:.3f}%)")

print("\nOutlet Temperature extremes:")
t_mean = df["OutletT"].mean()
t_std = df["OutletT"].std()
for sigma in [2, 3, 4]:
    lo = t_mean - sigma * t_std
    hi = t_mean + sigma * t_std
    cnt = ((df["OutletT"] < lo) | (df["OutletT"] > hi)).sum()
    print(f"  |OutletT - mean| > {sigma}σ  (outside [{lo:.1f}, {hi:.1f}]):  "
          f"{cnt:>8,} rows  ({100*cnt/n:.3f}%)")

# ── Overlap matrix ────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("=== Overlap Between Scenarios (strict) ===")
print(f"{'=' * 80}")
labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
masks = [s1_strict, s2_strict, s3_strict, s4_strict, s5_strict, s6_strict]
header = f"{'':>4s}" + "".join(f"{l:>8s}" for l in labels)
print(header)
for i, li in enumerate(labels):
    row = f"{li:>4s}"
    for j, lj in enumerate(labels):
        overlap = (masks[i] & masks[j]).sum()
        row += f"{overlap:>8,}"
    print(row)
