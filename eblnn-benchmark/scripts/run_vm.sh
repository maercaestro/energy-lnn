#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_vm.sh
# ---------------------------------------------------------------------------
# Launcher for the EB-LNN benchmark on a remote VM, designed to be run
# inside a tmux session and survive SSH disconnects.
#
# Workflow:
#   1. Activate the chosen Python env (set $EBLNN_PYTHON to override).
#   2. Make sure wandb is logged in (set $WANDB_API_KEY before launching).
#   3. Run the full benchmark sweep — resumable; rerun and it skips done runs.
#   4. Generate the comparison report and the publication figures.
#   5. Mirror everything to a timestamped log file under results/<run>/logs/.
#
# Typical usage on the VM:
#
#   tmux new -s eblnn
#   export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx
#   bash eblnn-benchmark/scripts/run_vm.sh
#   # detach with: Ctrl-b d
#   # reattach later with: tmux attach -t eblnn
#
# Re-run safety: if the script is killed mid-way, just relaunch — already
# completed (model, seed) pairs are detected via results.json and skipped.
# Pass FORCE=1 to ignore that and rerun from scratch.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${BENCH_DIR}")"
cd "${REPO_ROOT}"

# ---- env ----
PYTHON_BIN="${EBLNN_PYTHON:-python}"
RUN_NAME="${RUN_NAME:-default}"
CONFIG_PATH="${CONFIG_PATH:-${BENCH_DIR}/config/benchmark_config.yaml}"
EXTRA_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
    EXTRA_ARGS+=("--force")
fi
if [[ "${NO_WANDB:-0}" == "1" ]]; then
    EXTRA_ARGS+=("--no-wandb")
fi

# ---- log file ----
LOG_DIR="${BENCH_DIR}/results/${RUN_NAME}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/run-${TS}.log"

# Mirror everything to a log file (and still show on stdout when attached).
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "EB-LNN benchmark launcher"
echo "  repo root : ${REPO_ROOT}"
echo "  bench dir : ${BENCH_DIR}"
echo "  config    : ${CONFIG_PATH}"
echo "  run name  : ${RUN_NAME}"
echo "  python    : ${PYTHON_BIN}"
echo "  log file  : ${LOG_FILE}"
echo "  extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "  started   : $(date)"
echo "============================================================"

# ---- sanity ----
"${PYTHON_BIN}" -c "import torch, ncps; print('torch', torch.__version__, '| ncps', ncps.__version__)"
if [[ "${NO_WANDB:-0}" != "1" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "[warn] WANDB_API_KEY is not set; relying on ~/.netrc or wandb offline."
    fi
    "${PYTHON_BIN}" -c "import wandb; print('wandb', wandb.__version__)" || {
        echo "[warn] wandb not importable; continuing with --no-wandb."
        EXTRA_ARGS+=("--no-wandb")
    }
fi

# ---- 1. Run the benchmark sweep ----
echo ">>> Running benchmark sweep ..."
"${PYTHON_BIN}" "${BENCH_DIR}/scripts/run_benchmark.py" \
    --config "${CONFIG_PATH}" \
    "${EXTRA_ARGS[@]}"

# ---- 2. Aggregate ----
echo ">>> Building comparison report ..."
"${PYTHON_BIN}" "${BENCH_DIR}/scripts/compare_results.py" \
    --results_dir "${BENCH_DIR}/results/${RUN_NAME}"

# ---- 3. Plots ----
echo ">>> Generating plots ..."
"${PYTHON_BIN}" "${BENCH_DIR}/scripts/plot_results.py" \
    --results_dir "${BENCH_DIR}/results/${RUN_NAME}" || \
    echo "[warn] plot_results.py failed (non-fatal)."

echo "============================================================"
echo "Finished at $(date)"
echo "Outputs : ${BENCH_DIR}/results/${RUN_NAME}/"
echo "Log     : ${LOG_FILE}"
echo "============================================================"
