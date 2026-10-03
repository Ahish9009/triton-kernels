#!/usr/bin/env bash
#
# Sync this repo with the remote GPU server over rsync/ssh.
#
#   ./sync.sh push          mirror local -> remote (local is source of truth)
#   ./sync.sh pull          copy remote -> local (additive; brings back results)
#   ./sync.sh run [cmd...]  run a command in the remote repo inside the env
#                           (default: pytest -q)
#   ./sync.sh gpus          show GPU usage on the remote (to find a free one)
#
# Override the target with env vars:
#   REMOTE_HOST=ahishd@trinity.vision.cs.cmu.edu
#   REMOTE_DIR=triton-kernels   (relative to the remote home dir)
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ahishd@trinity.vision.cs.cmu.edu}"
REMOTE_DIR="${REMOTE_DIR:-triton-kernels}"
ENV_NAME="${ENV_NAME:-triton-kernels}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Respect .gitignore so caches, venvs, and build artifacts never transfer.
COMMON_OPTS=(-az --info=progress2 --exclude='.git/' --filter=':- .gitignore')

cmd="${1:-push}"
shift || true

case "${cmd}" in
  push)
    echo ">> push  ${SCRIPT_DIR}/  ->  ${REMOTE_HOST}:${REMOTE_DIR}/"
    rsync "${COMMON_OPTS[@]}" --delete \
      "${SCRIPT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
    ;;
  pull)
    echo ">> pull  ${REMOTE_HOST}:${REMOTE_DIR}/  ->  ${SCRIPT_DIR}/"
    rsync "${COMMON_OPTS[@]}" \
      "${REMOTE_HOST}:${REMOTE_DIR}/" "${SCRIPT_DIR}/"
    ;;
  run)
    remote_cmd="${*:-pytest -q}"
    echo ">> run on ${REMOTE_HOST}: ${remote_cmd}"
    ssh -t "${REMOTE_HOST}" \
      "cd '${REMOTE_DIR}' && conda run --no-capture-output -n '${ENV_NAME}' ${remote_cmd}"
    ;;
  gpus)
    echo ">> GPU usage on ${REMOTE_HOST}:"
    ssh "${REMOTE_HOST}" \
      "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
         --format=csv,nounits"
    ;;
  *)
    echo "usage: $0 [push|pull|run [cmd...]|gpus]" >&2
    exit 1
    ;;
esac
