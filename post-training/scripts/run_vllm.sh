#!/bin/bash
set -euo pipefail

# Usage: bash scripts/run_vllm.sh <examples/inference/*.yaml> [API_PORT]

if [ $# -lt 1 ]; then
  echo "Usage: $0 <inference_yaml> [API_PORT]"
  exit 1
fi

YAML="$1"
PORT="${2:-8000}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT/LLaMA-Factory"

API_PORT="$PORT" llamafactory-cli api "$YAML"
