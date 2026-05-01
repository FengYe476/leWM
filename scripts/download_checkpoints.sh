#!/usr/bin/env bash

# Download official LeWM checkpoints from the Hugging Face Hub.
# This follows the repo-documented pattern of using the Hugging Face CLI
# against model repositories.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"

log() {
  echo
  echo "==> $1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_hf_download_cmd() {
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
    return 0
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli"
    return 0
  fi

  return 1
}

download_repo() {
  local repo_id="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"
  log "Downloading ${repo_id} into ${out_dir}"
  "${HF_DOWNLOAD_CMD}" download "${repo_id}" --repo-type model --local-dir "${out_dir}"
}

log "Checking prerequisites"
HF_DOWNLOAD_CMD="$(detect_hf_download_cmd)" || {
  echo "Missing required command: hf or huggingface-cli" >&2
  exit 1
}

mkdir -p "${CHECKPOINT_ROOT}"

download_repo "quentinll/lewm-pusht" "${CHECKPOINT_ROOT}/lewm-pusht"
download_repo "quentinll/lewm-cube" "${CHECKPOINT_ROOT}/lewm-cube"

log "Checkpoint download complete"
echo "Stored checkpoints under: ${CHECKPOINT_ROOT}"
