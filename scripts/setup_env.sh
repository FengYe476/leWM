#!/usr/bin/env bash

# Set up the research environment for LeWM failure-mode analysis.
# The script prefers a verification-friendly install that works on both
# Linux GPU hosts and macOS ARM laptops.

set -euo pipefail

ENV_NAME="lewm-audit"
PYTHON_VERSION="3.11"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTORCH_LINUX_VERSION="${PYTORCH_LINUX_VERSION:-2.10.0}"
TORCHVISION_LINUX_VERSION="${TORCHVISION_LINUX_VERSION:-0.25.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"

log() {
  echo
  echo "==> $1"
}

warn() {
  echo "WARNING: $1" >&2
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

clone_or_update_repo() {
  local repo_url="$1"
  local dest_dir="$2"

  if [[ -d "${dest_dir}/.git" ]]; then
    log "Updating existing checkout: ${dest_dir}"
    git -C "${dest_dir}" pull --ff-only
  else
    log "Cloning ${repo_url} into ${dest_dir}"
    git clone "${repo_url}" "${dest_dir}"
  fi
}

install_editable_if_packaged() {
  local repo_dir="$1"
  local repo_name="$2"

  if [[ -f "${repo_dir}/pyproject.toml" || -f "${repo_dir}/setup.py" || -f "${repo_dir}/setup.cfg" ]]; then
    log "Installing ${repo_name} in editable mode"
    (
      cd "${repo_dir}"
      pip install -e .
    )
  else
    log "${repo_name} does not expose packaging metadata; leaving it as a source checkout only"
  fi
}

detect_platform() {
  local os_name arch_name
  os_name="$(uname -s)"
  arch_name="$(uname -m)"

  case "${os_name}:${arch_name}" in
    Darwin:arm64)
      echo "macos-arm64"
      ;;
    Linux:x86_64)
      echo "linux-x86_64"
      ;;
    *)
      echo "other"
      ;;
  esac
}

install_pytorch() {
  local platform="$1"

  case "${platform}" in
    macos-arm64)
      log "Installing PyTorch for macOS ARM (CPU / MPS-capable build, no CUDA)"
      if pip install torch torchvision; then
        return 0
      fi
      warn "Unpinned PyTorch install failed on macOS ARM; retrying with explicit stable pins"
      pip install "torch==${PYTORCH_LINUX_VERSION}" "torchvision==${TORCHVISION_LINUX_VERSION}"
      ;;
    linux-x86_64)
      log "Installing PyTorch for Linux x86_64 with CUDA 12.x wheels"
      echo "Using PyTorch wheel index: ${PYTORCH_CUDA_INDEX_URL}"
      if pip install \
        "torch==${PYTORCH_LINUX_VERSION}" \
        "torchvision==${TORCHVISION_LINUX_VERSION}" \
        --index-url "${PYTORCH_CUDA_INDEX_URL}"; then
        return 0
      fi
      warn "Pinned CUDA wheel install failed; retrying with unpinned CUDA wheels"
      if pip install torch torchvision --index-url "${PYTORCH_CUDA_INDEX_URL}"; then
        return 0
      fi
      warn "CUDA wheel install failed; falling back to default PyPI wheels"
      pip install torch torchvision
      ;;
    *)
      log "Installing PyTorch with default wheels for unsupported/unknown platform"
      pip install torch torchvision
      ;;
  esac
}

install_stable_worldmodel() {
  log "Installing stable-worldmodel"
  if pip install "stable-worldmodel[train,envs]"; then
    echo "stable-worldmodel extras variant: train,envs"
    return 0
  fi

  warn "Requested extra [envs] was not accepted; retrying with upstream [env]"
  if pip install "stable-worldmodel[train,env]"; then
    echo "stable-worldmodel extras variant: train,env"
    return 0
  fi

  warn "Full stable-worldmodel extras failed; installing base package plus verification-focused dependencies"
  pip install stable-worldmodel
  pip install pygame pymunk shapely gymnasium gymnasium-robotics opencv-python-headless minigrid ale-py
  echo "stable-worldmodel extras variant: base+manual-verification-fallback"
}

log "Checking prerequisites"
require_cmd conda
require_cmd git

PLATFORM="$(detect_platform)"
log "Detected platform: ${PLATFORM}"

log "Initializing conda for non-interactive shell usage"
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda environment ${ENV_NAME} already exists; reusing it"
else
  log "Creating conda environment ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

log "Activating conda environment ${ENV_NAME}"
conda activate "${ENV_NAME}"

log "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

install_pytorch "${PLATFORM}"
install_stable_worldmodel

log "Installing OGBench"
pip install ogbench

log "Installing stable-worldmodel runtime env dependencies"
# `stable-worldmodel[train,envs]` currently warns that `envs` is not a valid
# extra, so install the lightweight runtime pieces we need explicitly.
pip install pygame pymunk shapely

log "Installing analysis utilities"
pip install huggingface_hub matplotlib seaborn jupyter

mkdir -p "${THIRD_PARTY_DIR}"

log "Installing stable-pretraining from source"
clone_or_update_repo "https://github.com/galilai-group/stable-pretraining.git" "${THIRD_PARTY_DIR}/stable-pretraining"
(
  cd "${THIRD_PARTY_DIR}/stable-pretraining"
  pip install -e .
)

log "Cloning LeWM source repo"
clone_or_update_repo "https://github.com/lucas-maes/le-wm.git" "${THIRD_PARTY_DIR}/le-wm"
install_editable_if_packaged "${THIRD_PARTY_DIR}/le-wm" "LeWM"

log "Environment setup complete"
echo "Environment name: ${ENV_NAME}"
echo "Project root: ${PROJECT_ROOT}"
echo "Third-party sources: ${THIRD_PARTY_DIR}"
