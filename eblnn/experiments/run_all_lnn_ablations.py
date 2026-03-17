"""
run_all_lnn_ablations.py
========================
Launch every LNN ablation experiment sequentially with W&B logging.
Designed for long-running VM sessions via ``nohup``.

Usage
-----
    cd eblnn

    # Run all
    nohup python -u experiments/run_all_lnn_ablations.py > lnn_ablation.log 2>&1 &
    tail -f lnn_ablation.log

    # Run a specific axis only
    nohup python -u experiments/run_all_lnn_ablations.py --axis A > lnn_ablation_A.log 2>&1 &

    # Skip completed experiments (resume after crash)
    nohup python -u experiments/run_all_lnn_ablations.py --skip_existing > lnn_ablation.log 2>&1 &

    # Disable W&B
    nohup python -u experiments/run_all_lnn_ablations.py --no_wandb > lnn_ablation.log 2>&1 &

    # Dry run — preview commands
    python experiments/run_all_lnn_ablations.py --dry_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "lnn_ablation_config.yaml"
RUNNER = ROOT / "experiments" / "run_lnn_ablation.py"


def load_experiments(axis: str | None = None) -> list[str]:
    """Return experiment names, optionally filtered by axis prefix."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    names = list(cfg["experiments"].keys())
    if axis:
        names = [n for n in names if n.startswith(axis)]
    return sorted(names)


def main() -> None:
    p = argparse.ArgumentParser(description="Run all LNN ablation experiments")
    p.add_argument("--axis", type=str, default=None,
                   help="Only run experiments starting with this prefix (A/B/C/D/E)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip experiments that already have results.json")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Run each experiment with these seeds (e.g. --seeds 42 123 456)")
    args = p.parse_args()

    experiments = load_experiments(args.axis)
    seeds = args.seeds or [None]   # None = use config default (42)
    total_runs = len(experiments) * len(seeds)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wandb_status = "OFF" if args.no_wandb else "ON"

    print(f"\n{'═' * 65}", flush=True)
    print(f"  LNN ABLATION BATCH  —  {len(experiments)} experiments × {len(seeds)} seed(s) = {total_runs} runs", flush=True)
    if args.seeds:
        print(f"  Seeds   : {seeds}", flush=True)
    print(f"  Started : {ts}", flush=True)
    print(f"  W&B     : {wandb_status}", flush=True)
    print(f"{'═' * 65}\n", flush=True)
    for i, name in enumerate(experiments, 1):
        print(f"  {i:2d}. {name}", flush=True)
    print(flush=True)

    if args.dry_run:
        print("[dry run] Commands that would be executed:\n", flush=True)
        for name in experiments:
            for seed in seeds:
                cmd = _build_cmd(name, args, seed)
                print(f"  {' '.join(cmd)}\n", flush=True)
        return

    results_summary: list[dict] = []
    t0_all = time.time()
    run_idx = 0

    for name in experiments:
        for seed in seeds:
            run_idx += 1
            seed_tag = f" (seed={seed})" if seed is not None else ""
            seed_suffix = f"_seed{seed}" if seed is not None else ""

            if args.skip_existing:
                rpath = ROOT / "results" / "lnn_ablation" / (name + seed_suffix) / "results.json"
                if rpath.exists():
                    print(f"\n[{run_idx}/{total_runs}] SKIP {name}{seed_tag} (results.json exists)", flush=True)
                    continue

            ts_now = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'─' * 65}", flush=True)
            print(f"  [{run_idx}/{total_runs}]  {name}{seed_tag}  (started {ts_now})", flush=True)
            print(f"{'─' * 65}", flush=True)

            cmd = _build_cmd(name, args, seed)
            t0 = time.time()
            ret = subprocess.run(cmd, cwd=str(ROOT))
            elapsed = time.time() - t0

            results_summary.append({
                "experiment": name,
                "seed": seed,
                "return_code": ret.returncode,
                "wall_time_sec": round(elapsed, 1),
            })

            status = "✅" if ret.returncode == 0 else "❌"
            print(f"\n{status}  {name}{seed_tag}  ({elapsed:.0f}s)", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t0_all
    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'═' * 65}", flush=True)
    print(f"  BATCH COMPLETE  —  {len(results_summary)} experiments", flush=True)
    print(f"  Finished : {ts_end}", flush=True)
    print(f"  Total wall time : {total_time:.0f}s  ({total_time / 60:.1f} min)", flush=True)
    print(f"{'═' * 65}\n", flush=True)

    ok = sum(1 for r in results_summary if r["return_code"] == 0)
    fail = len(results_summary) - ok
    print(f"  ✅ Passed: {ok}    ❌ Failed: {fail}\n", flush=True)

    if fail:
        print("  Failed experiments:", flush=True)
        for r in results_summary:
            if r["return_code"] != 0:
                print(f"    - {r['experiment']}  (exit {r['return_code']})", flush=True)
        print(flush=True)


def _build_cmd(name: str, args, seed: int | None = None) -> list[str]:
    cmd = [sys.executable, "-u", str(RUNNER), "--experiment", name]
    if args.no_wandb:
        cmd.append("--no_wandb")
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


if __name__ == "__main__":
    main()
