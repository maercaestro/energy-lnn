#!/usr/bin/env bash
# Run inside tmux on a VM. The W&B agent can be relaunched after interruption.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EBLNN_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${EBLNN_DIR}")"
PYTHON_BIN="${EBLNN_PYTHON:-python}"
COUNT="${COUNT:-30}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the operating-system environment before launching the sweep.}"
cd "${EBLNN_DIR}"

mkdir -p results/ebm_tuning/logs
LOG_FILE="results/ebm_tuning/logs/tuning-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "EB-LNN EBM/CD tuning"
echo "repo: ${REPO_ROOT}"
echo "trials: ${COUNT}"
echo "log: ${EBLNN_DIR}/${LOG_FILE}"
"${PYTHON_BIN}" -c "import torch, ncps, wandb; print('torch', torch.__version__, '| ncps', ncps.__version__, '| wandb', wandb.__version__)"

SWEEP_ID="$(${PYTHON_BIN} -m wandb sweep "${SCRIPT_DIR}/sweep.yaml" | tee /dev/stderr | sed -n 's/.*wandb agent \([^ ]*\).*/\1/p' | tail -n 1)"
if [[ -z "${SWEEP_ID}" ]]; then
    echo "Could not obtain the W&B sweep ID. Create it manually with:"
    echo "${PYTHON_BIN} -m wandb sweep ${SCRIPT_DIR}/sweep.yaml"
    exit 1
fi

"${PYTHON_BIN}" -m wandb agent --count "${COUNT}" "${SWEEP_ID}"