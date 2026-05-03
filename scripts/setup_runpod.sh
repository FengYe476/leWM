#!/usr/bin/env bash

# One-shot setup for a fresh RunPod instance:
# - no conda
# - system Python 3.12 with pip
# - CUDA 12.8
# - PyTorch / torchvision already installed

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
CONVERTED_ROOT="${CHECKPOINT_ROOT}/converted"
VERIFY_SENTINEL="${CONVERTED_ROOT}/.verify_checkpoint.ok"
CACHE_ROOT="${PROJECT_ROOT}/stablewm_cache"
PYTHON_BIN="${PYTHON_BIN:-python}"

STABLE_PRETRAINING_REPO="https://github.com/galilai-group/stable-pretraining.git"
LEWM_REPO="https://github.com/lucas-maes/le-wm.git"

STEP_NAMES=()
STEP_STATUSES=()
STEP_DETAILS=()
FAILED_COUNT=0

log() {
  echo
  echo "==> $1"
}

info() {
  echo "    $1"
}

warn() {
  echo "WARNING: $1" >&2
}

record_step() {
  local name="$1"
  local status="$2"
  local detail="${3:-}"

  STEP_NAMES+=("${name}")
  STEP_STATUSES+=("${status}")
  STEP_DETAILS+=("${detail}")

  if [[ "${status}" != "OK" ]]; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
}

run_step() {
  local name="$1"
  shift

  log "${name}"
  if (set -euo pipefail; "$@"); then
    record_step "${name}" "OK" ""
  else
    local exit_code=$?
    record_step "${name}" "FAILED" "exit ${exit_code}"
    warn "${name} failed with exit code ${exit_code}; continuing so the final summary is complete."
  fi
}

print_summary() {
  echo
  echo "========================================"
  echo "RunPod setup summary"
  echo "========================================"

  local i
  for ((i = 0; i < ${#STEP_NAMES[@]}; i++)); do
    if [[ -n "${STEP_DETAILS[$i]}" ]]; then
      printf "[%s] %s (%s)\n" "${STEP_STATUSES[$i]}" "${STEP_NAMES[$i]}" "${STEP_DETAILS[$i]}"
    else
      printf "[%s] %s\n" "${STEP_STATUSES[$i]}" "${STEP_NAMES[$i]}"
    fi
  done

  echo
  echo "Project root: ${PROJECT_ROOT}"
  echo "Checkpoints: ${CHECKPOINT_ROOT}"
  echo "Datasets:"
  echo "  ${CACHE_ROOT}/pusht_expert_train.h5"
  echo "  ${CACHE_ROOT}/ogbench/cube_single_expert.h5"
  echo "Third-party sources: ${THIRD_PARTY_DIR}"

  if [[ "${FAILED_COUNT}" -eq 0 ]]; then
    echo
    echo "All setup steps completed successfully."
  else
    echo
    echo "${FAILED_COUNT} setup step(s) failed. Review the messages above and rerun this script after fixing them."
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    return 1
  fi
}

ensure_apt_package() {
  local cmd="$1"
  local package="$2"

  if command -v "${cmd}" >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}; apt-get is unavailable, so install ${package} manually." >&2
    return 1
  fi

  info "Installing missing system package ${package}"
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${package}"
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${package}"
  else
    echo "Missing required command: ${cmd}; sudo is unavailable, so install ${package} manually." >&2
    return 1
  fi
}

clone_or_update_repo() {
  local repo_url="$1"
  local dest_dir="$2"

  if [[ -d "${dest_dir}/.git" ]]; then
    info "Updating existing checkout: ${dest_dir}"
    git -C "${dest_dir}" pull --ff-only
  elif [[ -e "${dest_dir}" ]]; then
    echo "${dest_dir} exists but is not a git checkout; move it aside or remove it before rerunning." >&2
    return 1
  else
    info "Cloning ${repo_url} into ${dest_dir}"
    git clone "${repo_url}" "${dest_dir}"
  fi
}

install_editable_if_packaged() {
  local repo_dir="$1"
  local repo_name="$2"

  if [[ -f "${repo_dir}/pyproject.toml" || -f "${repo_dir}/setup.py" || -f "${repo_dir}/setup.cfg" ]]; then
    info "Installing ${repo_name} in editable mode"
    (
      cd "${repo_dir}"
      "${PYTHON_BIN}" -m pip install -e .
    )
  else
    info "${repo_name} has no packaging metadata; source checkout is enough for this repo's sys.path imports"
  fi
}

install_pip_dependencies() {
  require_command "${PYTHON_BIN}"

  info "Python version:"
  "${PYTHON_BIN}" --version

  info "Confirming pre-installed torch / torchvision; this script will not install either package"
  "${PYTHON_BIN}" - <<'PY'
import importlib

for name in ("torch", "torchvision"):
    module = importlib.import_module(name)
    print(f"{name}=={getattr(module, '__version__', 'unknown')}")
PY

  info "Upgrading pip/setuptools/wheel"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

  info "Installing Python dependencies, excluding torch and torchvision"
  "${PYTHON_BIN}" -m pip install \
    "stable-worldmodel[train]" \
    ogbench \
    pygame \
    pymunk \
    shapely \
    gymnasium \
    opencv-python-headless \
    matplotlib \
    seaborn \
    huggingface_hub \
    scipy \
    scikit-learn

  require_command hf
}

install_third_party_repos() {
  require_command git
  mkdir -p "${THIRD_PARTY_DIR}"

  clone_or_update_repo "${STABLE_PRETRAINING_REPO}" "${THIRD_PARTY_DIR}/stable-pretraining"
  install_editable_if_packaged "${THIRD_PARTY_DIR}/stable-pretraining" "stable-pretraining"

  clone_or_update_repo "${LEWM_REPO}" "${THIRD_PARTY_DIR}/le-wm"
  install_editable_if_packaged "${THIRD_PARTY_DIR}/le-wm" "LeWM"
}

checkpoint_is_complete() {
  local ckpt_dir="$1"
  [[ -s "${ckpt_dir}/config.json" && -s "${ckpt_dir}/weights.pt" ]]
}

download_checkpoint_repo() {
  local repo_id="$1"
  local out_dir="$2"

  if checkpoint_is_complete "${out_dir}"; then
    info "${repo_id} checkpoint already present at ${out_dir}; skipping download"
    return 0
  fi

  mkdir -p "${out_dir}"
  info "Downloading ${repo_id} model repo into ${out_dir}"
  hf download "${repo_id}" --repo-type model --local-dir "${out_dir}"

  if ! checkpoint_is_complete "${out_dir}"; then
    echo "Downloaded ${repo_id}, but config.json and/or weights.pt is missing in ${out_dir}." >&2
    return 1
  fi
}

download_checkpoints() {
  require_command hf
  mkdir -p "${CHECKPOINT_ROOT}"

  download_checkpoint_repo "quentinll/lewm-pusht" "${CHECKPOINT_ROOT}/lewm-pusht"
  download_checkpoint_repo "quentinll/lewm-cube" "${CHECKPOINT_ROOT}/lewm-cube"
}

verify_checkpoints() {
  local pusht_object="${CONVERTED_ROOT}/lewm-pusht/lewm_object.ckpt"
  local cube_object="${CONVERTED_ROOT}/lewm-cube/lewm_object.ckpt"

  if [[ -s "${VERIFY_SENTINEL}" && -s "${pusht_object}" && -s "${cube_object}" ]]; then
    local newest_input
    newest_input="$(
      find \
        "${CHECKPOINT_ROOT}/lewm-pusht" \
        "${CHECKPOINT_ROOT}/lewm-cube" \
        "${PROJECT_ROOT}/scripts/verify_checkpoint.py" \
        -type f \
        \( -name "config.json" -o -name "weights.pt" -o -name "verify_checkpoint.py" \) \
        -newer "${VERIFY_SENTINEL}" \
        -print \
        -quit
    )"
    if [[ -z "${newest_input}" ]]; then
      info "Converted checkpoints are already verified; skipping verify_checkpoint.py"
      return 0
    fi
    info "Verifier inputs changed since last successful run; re-running verification"
  fi

  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" scripts/verify_checkpoint.py
  )
  mkdir -p "${CONVERTED_ROOT}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"${VERIFY_SENTINEL}"
}

find_zst_file() {
  local search_dir="$1"
  shift

  local expected_name
  for expected_name in "$@"; do
    local match
    match="$(find "${search_dir}" -type f -name "${expected_name}" -print -quit)"
    if [[ -n "${match}" ]]; then
      echo "${match}"
      return 0
    fi
  done
}

download_and_decompress_dataset() {
  local repo_id="$1"
  local local_dir="$2"
  local h5_path="$3"
  shift 3
  local archive_names=("$@")

  if [[ -s "${h5_path}" ]]; then
    info "${h5_path} already exists; skipping dataset download and decompression"
    return 0
  fi

  mkdir -p "${local_dir}"

  local zst_path
  zst_path="$(find_zst_file "${local_dir}" "${archive_names[@]}")"

  if [[ -z "${zst_path}" ]]; then
    info "Downloading ${repo_id} dataset .zst files into ${local_dir}"
    hf download "${repo_id}" --repo-type dataset --include "*.zst" --local-dir "${local_dir}"
    zst_path="$(find_zst_file "${local_dir}" "${archive_names[@]}")"
  else
    info "Found existing compressed dataset: ${zst_path}"
  fi

  if [[ -z "${zst_path}" || ! -s "${zst_path}" ]]; then
    echo "Could not find expected dataset archive under ${local_dir} after download. Looked for: ${archive_names[*]}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${h5_path}")"

  if [[ "${zst_path}" == *.tar.zst ]]; then
    require_command tar
    local tar_path="${zst_path%.zst}"
    info "Decompressing tar archive ${zst_path}"
    zstd -d -f "${zst_path}" -o "${tar_path}"
    info "Extracting ${tar_path} into ${local_dir}"
    tar -xf "${tar_path}" -C "${local_dir}"
    rm -f "${tar_path}"

    if [[ ! -s "${h5_path}" ]]; then
      local extracted_h5
      extracted_h5="$(find "${local_dir}" -type f -name "$(basename "${h5_path}")" -print -quit)"
      if [[ -n "${extracted_h5}" && "${extracted_h5}" != "${h5_path}" ]]; then
        info "Moving extracted HDF5 into canonical location: ${h5_path}"
        mv -f "${extracted_h5}" "${h5_path}"
      fi
    fi
  else
    info "Decompressing ${zst_path} to ${h5_path}"
    zstd -d -f "${zst_path}" -o "${h5_path}"
  fi

  if [[ ! -s "${h5_path}" ]]; then
    echo "Decompression did not produce expected file: ${h5_path}" >&2
    return 1
  fi

  info "Removing compressed archive to save disk space: ${zst_path}"
  rm -f "${zst_path}"
}

download_datasets() {
  require_command hf
  ensure_apt_package zstd zstd

  download_and_decompress_dataset \
    "quentinll/lewm-pusht" \
    "${CACHE_ROOT}" \
    "${CACHE_ROOT}/pusht_expert_train.h5" \
    "pusht_expert_train.h5.zst" \
    "pusht_expert_train.tar.zst"

  download_and_decompress_dataset \
    "quentinll/lewm-cube" \
    "${CACHE_ROOT}/ogbench" \
    "${CACHE_ROOT}/ogbench/cube_single_expert.h5" \
    "cube_single_expert.h5.zst" \
    "cube_single_expert.tar.zst"
}

main() {
  echo "RunPod LeWM setup"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Python command: ${PYTHON_BIN}"

  run_step "Install pip dependencies" install_pip_dependencies
  run_step "Clone and install third-party repositories" install_third_party_repos
  run_step "Download LeWM checkpoints" download_checkpoints
  run_step "Convert and verify checkpoints" verify_checkpoints
  run_step "Download and decompress datasets" download_datasets

  print_summary
  [[ "${FAILED_COUNT}" -eq 0 ]]
}

main "$@"
