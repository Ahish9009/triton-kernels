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
# Files sync to the LOGIN node (shared NFS home -> visible on every node).
# Commands run on the COMPUTE node, reached by jumping through the login node.
#
# Override the targets with env vars:
#   REMOTE_HOST=ahishd@trinity.vision.cs.cmu.edu   (login node, for file sync)
#   COMPUTE_HOST=ahishd@trinity-0-3                 (compute node, for run/gpus)
#   REMOTE_DIR=triton-kernels                       (relative to the remote home dir)
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ahishd@trinity.vision.cs.cmu.edu}"
COMPUTE_HOST="${COMPUTE_HOST:-ahishd@trinity-0-3}"
REMOTE_DIR="${REMOTE_DIR:-triton-kernels}"
ENV_NAME="${ENV_NAME:-triton-kernels}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# push: mirror local -> remote, respecting .gitignore so caches / venvs /
# build artifacts never transfer.
PUSH_OPTS=(-az --info=progress2 --exclude='.git/' --filter=':- .gitignore')

# pull: bring results back (incl. gitignored benchmark_outputs/), but skip
# caches and .git so they don't pollute the local tree.
PULL_OPTS=(-az --info=progress2
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='*.pyc'
)

cmd="${1:-push}"
shift || true

case "${cmd}" in
  push)
    echo ">> push  ${SCRIPT_DIR}/  ->  ${REMOTE_HOST}:${REMOTE_DIR}/"
    rsync "${PUSH_OPTS[@]}" --delete \
      "${SCRIPT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
    ;;
  pull)
    echo ">> pull  ${REMOTE_HOST}:${REMOTE_DIR}/  ->  ${SCRIPT_DIR}/"
    rsync "${PULL_OPTS[@]}" \
      "${REMOTE_HOST}:${REMOTE_DIR}/" "${SCRIPT_DIR}/"
    ;;
  run)
    remote_cmd="${*:-pytest -q}"
    echo ">> run on ${COMPUTE_HOST} (via ${REMOTE_HOST}): ${remote_cmd}"
    ssh -t -J "${REMOTE_HOST}" "${COMPUTE_HOST}" \
      "cd '${REMOTE_DIR}' && conda run --no-capture-output -n '${ENV_NAME}' ${remote_cmd}"
    ;;
  gpus)
    echo ">> GPU usage on ${COMPUTE_HOST} (via ${REMOTE_HOST}):"
    ssh -J "${REMOTE_HOST}" "${COMPUTE_HOST}" \
      "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
         --format=csv,nounits"
    ;;
  *)
    echo "usage: $0 [push|pull|run [cmd...]|gpus]" >&2
    exit 1
    ;;
esac
