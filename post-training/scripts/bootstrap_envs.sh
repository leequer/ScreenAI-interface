#!/bin/bash
set -euo pipefail

# bootstrap_envs.sh: set up conda envs and data for this repo
# Usage:
#   bash scripts/bootstrap_envs.sh            # install all (rl, vllm, data, sft)
#   bash scripts/bootstrap_envs.sh rl         # only RL env
#   bash scripts/bootstrap_envs.sh vllm       # only vLLM env
#   bash scripts/bootstrap_envs.sh sft        # only SFT env
#   bash scripts/bootstrap_envs.sh data       # only fetch demo data

# Configurable via env vars
CONDA_ROOT=${CONDA_ROOT:-"$HOME/miniconda3"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# Resolve conda binary
if [ -x "$CONDA_ROOT/bin/conda" ]; then
  CONDA_BIN="$CONDA_ROOT/bin/conda"
else
  CONDA_BIN="conda"
fi

os_is_linux() {
  [[ "$(uname -s)" == "Linux" ]]
}

concat_parts_if_needed() {
  local name="$1"
  local parts=("$REPO_ROOT/conda_envs/${name}.tar.gz.part.")
  shopt -s nullglob
  local matches=("$REPO_ROOT/conda_envs/${name}.tar.gz.part."*)
  shopt -u nullglob
  if [ ${#matches[@]} -gt 0 ]; then
    echo "Concatenating ${name} env parts..."
    cat "$REPO_ROOT/conda_envs/${name}.tar.gz.part."* > "$REPO_ROOT/conda_envs/${name}.tar.gz"
  fi
}

extract_env_archive() {
  local name="$1"
  local env_path="$CONDA_ROOT/envs/$name"
  local archive="$REPO_ROOT/conda_envs/${name}.tar.gz"
  if [ -f "$archive" ]; then
    echo "Installing $name environment from archive..."
    mkdir -p "$env_path"
    tar -xzvf "$archive" -C "$env_path"
    if [ -x "$env_path/bin/conda-unpack" ]; then
      "$env_path/bin/conda-unpack"
    fi
    return 0
  fi
  return 1
}

install_rl() {
  concat_parts_if_needed rl || true
  extract_env_archive rl || true
  echo "Configuring RL environment (pip installs)..."
  pushd "$REPO_ROOT/VLM-R1/src/open-r1-multimodal" >/dev/null
  "$CONDA_BIN" run -n rl pip install -e ".[dev]"
  "$CONDA_BIN" run -n rl playwright install
  if os_is_linux; then
    "$CONDA_BIN" run -n rl playwright install-deps
  fi
  popd >/dev/null
  echo "RL environment ready."
}

install_vllm() {
  concat_parts_if_needed vllm || true
  extract_env_archive vllm || true
  echo "Configuring vLLM environment (metrics only editable LLaMA-Factory)..."
  pushd "$REPO_ROOT/LLaMA-Factory" >/dev/null
  "$CONDA_BIN" run -n vllm pip install -e ".[metrics]"
  popd >/dev/null
  echo "vLLM environment ready."
}

install_sft() {
  echo "Configuring SFT environment..."
  pushd "$REPO_ROOT/LLaMA-Factory" >/dev/null
  "$CONDA_BIN" run -n sft pip install -e ".[metrics,torch]"
  "$CONDA_BIN" run -n sft pip install deepspeed
  popd >/dev/null
  echo "SFT environment ready."
}

fetch_data() {
  echo "Fetching example data for LLaMA-Factory..."
  pushd "$REPO_ROOT/LLaMA-Factory/data" >/dev/null
  bash get_data.sh
  popd >/dev/null
  echo "Data prepared."
}

run_target() {
  case "${1:-all}" in
    rl) install_rl ;;
    vllm) install_vllm ;;
    sft) install_sft ;;
    data) fetch_data ;;
    all)
      install_rl
      install_vllm
      fetch_data
      install_sft
      ;;
    *)
      echo "Usage: $0 [rl|vllm|sft|data|all]" >&2
      exit 1
      ;;
  esac
}

run_target "${1:-all}"
