"""
Phase 1.1 — Data Cleaning Pipeline
====================================
Cleans DataCollected.csv → cleaned_furnace_data.csv

Cleaning steps:
  1. Replace historian bad-data tags with NaN
  2. Coerce all columns to float64
  3. Remove shutdown rows (InletFlow<20, OutletT<200, FGFlow<50)
  4. Remove 2-hour post-restart ramp-up after each shutdown block
  5. Remove O2 calibration spikes (ExcessO2 > 15)
  6. Remove impossible damper values (OP_Damper < 0 or > 100)
  7. Drop StackTemp (frozen sensor)
  8. Interpolate NaN gaps ≤ 10 min; mark gaps > 10 min as sequence breaks
  9. Remove 3σ outliers per column (on normal-operation data only)
 10. Save cleaned CSV + cleaning report

Usage:
    cd energy-lnn/dataset
    python clean_furnace_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent / "DataCollected.csv"
OUT_PATH = Path(__file__).parent / "cleaned_furnace_data.csv"
REPORT_PATH = Path(__file__).parent / "cleaning_report.txt"

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
SHUTDOWN_RULES = {
    "InletFlow":  ("lt", 20),
    "OutletT":    ("lt", 200),
    "FGFlow":     ("lt", 50),
}
POST_RESTART_MINUTES = 120          # 2 hours after each shutdown ends
O2_CALIB_THRESHOLD = 15.0           # ExcessO2 > 15 = calibration flush
DAMPER_RANGE = (0, 100)             # physical limits for OP_Damper
INTERP_MAX_GAP = 10                 # interpolate gaps ≤ 10 rows (minutes)
OUTLIER_SIGMA = 3.0                 # 3σ outlier fence (applied per column)
BAD_DATA_STRINGS = [
    "[-11059] No Good Data For Calculation",
    "No Data",
    "Bad",
    "Calc Failed",
]

# Columns to keep (drop StackTemp)
KEEP_COLS = [
    "Date", "InletT", "OutletT", "DraftP", "OP_Damper", "InletFlow",
    "ExcessO2", "FGFlow", "FGPressure", "Bridgewall",
]


def log(msg: str, report_lines: list):
    print(msg)
    report_lines.append(msg)


def main():
    report = []
    log("=" * 60, report)
    log("Furnace Data Cleaning Pipeline", report)
    log("=" * 60, report)

    # ── 1. Load ──────────────────────────────────────────────────────
    df = pd.read_csv(RAW_PATH, low_memory=False)
    n_raw = len(df)
    log(f"\n[1] Loaded {n_raw:,} rows, {df.shape[1]} columns", report)
    log(f"    Columns: {df.columns.tolist()}", report)

    # ── 2. Replace bad-data strings → NaN ────────────────────────────
    replaced = 0
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].isin(BAD_DATA_STRINGS)
            n = mask.sum()
            if n > 0:
                df.loc[mask, col] = np.nan
                replaced += n
                log(f"    {col}: replaced {n} bad-data tags → NaN", report)
    log(f"\n[2] Bad-data tags replaced: {replaced}", report)

    # ── 3. Coerce all to float64 ─────────────────────────────────────
    for col in df.columns:
        if col == "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    remaining_nan = df.isnull().sum().sum()
    log(f"\n[3] Coerced to numeric. Total NaN cells: {remaining_nan}", report)

    # ── 4. Drop StackTemp (frozen) ───────────────────────────────────
    if "StackTemp" in df.columns:
        df.drop(columns=["StackTemp"], inplace=True)
        log("\n[4] Dropped StackTemp (frozen at ~309°C)", report)

    # Keep only expected columns
    existing_keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[existing_keep].copy()
    log(f"    Retained columns: {df.columns.tolist()}", report)

    # ── 5. Remove shutdown rows ──────────────────────────────────────
    shutdown_mask = pd.Series(False, index=df.index)
    for col, (op, val) in SHUTDOWN_RULES.items():
        if col in df.columns:
            if op == "lt":
                m = df[col] < val
            else:
                m = df[col] > val
            shutdown_mask |= m
            log(f"    {col} {op} {val}: {m.sum():,} rows flagged", report)

    n_shutdown = shutdown_mask.sum()
    log(f"\n[5] Shutdown rows: {n_shutdown:,} ({n_shutdown/n_raw*100:.1f}%)", report)

    # ── 6. Remove post-restart ramp-up (2 hrs after each shutdown block) ──
    # Find transitions: shutdown→normal
    shutdown_int = shutdown_mask.astype(int)
    restart_points = (shutdown_int.diff() == -1)  # 1→0 = restart
    n_restarts = restart_points.sum()

    post_restart_mask = pd.Series(False, index=df.index)
    restart_idxs = df.index[restart_points].tolist()
    for idx in restart_idxs:
        end_idx = min(idx + POST_RESTART_MINUTES, df.index[-1])
        post_restart_mask.loc[idx:end_idx] = True

    # Don't double-count rows already in shutdown
    post_restart_only = post_restart_mask & ~shutdown_mask
    n_post_restart = post_restart_only.sum()
    log(f"\n[6] Detected {n_restarts} restart events", report)
    log(f"    Post-restart ramp-up rows removed: {n_post_restart:,}", report)

    # Combine removal mask
    remove_mask = shutdown_mask | post_restart_mask

    # ── 7. Remove O2 calibration spikes ──────────────────────────────
    o2_calib = df["ExcessO2"] > O2_CALIB_THRESHOLD
    n_o2_calib = (o2_calib & ~remove_mask).sum()
    remove_mask |= o2_calib
    log(f"\n[7] ExcessO2 > {O2_CALIB_THRESHOLD}: {n_o2_calib:,} additional rows", report)

    # ── 8. Remove impossible damper values ────────────────────────────
    damper_bad = (df["OP_Damper"] < DAMPER_RANGE[0]) | (df["OP_Damper"] > DAMPER_RANGE[1])
    n_damper = (damper_bad & ~remove_mask).sum()
    remove_mask |= damper_bad
    log(f"\n[8] OP_Damper outside [{DAMPER_RANGE[0]}, {DAMPER_RANGE[1]}]: "
        f"{n_damper:,} additional rows", report)

    # ── Apply removals ───────────────────────────────────────────────
    n_removed = remove_mask.sum()
    df_clean = df[~remove_mask].copy()
    log(f"\n[*] Total rows removed: {n_removed:,} ({n_removed/n_raw*100:.1f}%)", report)
    log(f"    Remaining: {len(df_clean):,}", report)

    # ── 9. Interpolate small NaN gaps, flag large ones ───────────────
    # After removals, some NaN cells may remain from the coercion
    nan_before = df_clean.isnull().sum()
    numeric_cols = [c for c in df_clean.columns if c != "Date"]

    for col in numeric_cols:
        # Find NaN runs
        is_nan = df_clean[col].isnull()
        if not is_nan.any():
            continue

        # Group consecutive NaNs
        nan_groups = is_nan.ne(is_nan.shift()).cumsum()
        nan_run_lengths = is_nan.groupby(nan_groups).transform("sum")

        # Interpolate short gaps
        short_gap = is_nan & (nan_run_lengths <= INTERP_MAX_GAP)
        if short_gap.any():
            df_clean[col] = df_clean[col].interpolate(method="linear", limit=INTERP_MAX_GAP)

    nan_after = df_clean.isnull().sum()
    nan_interpolated = (nan_before - nan_after).clip(lower=0)
    log(f"\n[9] Interpolated NaN gaps ≤ {INTERP_MAX_GAP} min:", report)
    for col in numeric_cols:
        if nan_interpolated.get(col, 0) > 0:
            log(f"    {col}: {nan_interpolated[col]} cells interpolated", report)

    # Drop any remaining rows with NaN (large gaps)
    n_nan_remaining = df_clean.isnull().any(axis=1).sum()
    if n_nan_remaining > 0:
        df_clean = df_clean.dropna()
        log(f"    Dropped {n_nan_remaining} rows with un-interpolatable NaN gaps", report)

    # ── 10. Remove 3σ outliers (per column, on clean data) ───────────
    n_before_outlier = len(df_clean)
    outlier_mask = pd.Series(False, index=df_clean.index)

    for col in numeric_cols:
        mu = df_clean[col].mean()
        sigma = df_clean[col].std()
        lo = mu - OUTLIER_SIGMA * sigma
        hi = mu + OUTLIER_SIGMA * sigma
        col_outlier = (df_clean[col] < lo) | (df_clean[col] > hi)
        n_col_outlier = col_outlier.sum()
        if n_col_outlier > 0:
            outlier_mask |= col_outlier
            log(f"    {col}: {n_col_outlier:,} outlier rows "
                f"(range [{lo:.1f}, {hi:.1f}])", report)

    df_clean = df_clean[~outlier_mask].copy()
    n_outlier_removed = n_before_outlier - len(df_clean)
    log(f"\n[10] 3σ outliers removed: {n_outlier_removed:,}", report)

    # ── 11. Contiguous block analysis ────────────────────────────────
    # After all cleaning, the Date column has gaps.
    # Identify contiguous blocks (consecutive Date values).
    date_diff = df_clean["Date"].diff()
    block_break = date_diff > 1
    block_id = block_break.cumsum()
    block_sizes = block_id.value_counts().sort_index()

    n_blocks = len(block_sizes)
    longest_block = block_sizes.max()
    median_block = int(block_sizes.median())
    log(f"\n[11] Contiguous block analysis:", report)
    log(f"    Total blocks: {n_blocks}", report)
    log(f"    Longest block: {longest_block:,} min ({longest_block/60:.1f} hrs)", report)
    log(f"    Median block: {median_block:,} min ({median_block/60:.1f} hrs)", report)
    log(f"    Blocks ≥ 60 min:  {(block_sizes >= 60).sum()}", report)
    log(f"    Blocks ≥ 240 min: {(block_sizes >= 240).sum()}", report)
    log(f"    Blocks ≥ 1440 min (1 day): {(block_sizes >= 1440).sum()}", report)

    # ── 12. Final summary & save ─────────────────────────────────────
    log(f"\n{'=' * 60}", report)
    log(f"FINAL DATASET", report)
    log(f"{'=' * 60}", report)
    log(f"  Rows:    {len(df_clean):,}  (from {n_raw:,} raw, {len(df_clean)/n_raw*100:.1f}% retained)", report)
    log(f"  Columns: {df_clean.columns.tolist()}", report)
    log(f"  NaN:     {df_clean.isnull().sum().sum()}", report)
    log(f"  dtypes:  all float64 (except Date: int)", report)
    log("", report)
    log("Column statistics:", report)
    log(df_clean.describe().to_string(), report)

    # Save
    df_clean.to_csv(OUT_PATH, index=False)
    log(f"\nSaved → {OUT_PATH}", report)

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))
    log(f"Report → {REPORT_PATH}", report)


if __name__ == "__main__":
    main()
