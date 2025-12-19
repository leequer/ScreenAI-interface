#!/bin/bash
set -euo pipefail

# Usage: bash scripts/run_sft.sh <train_yaml_or_cmd>
# If a YAML path is passed, it will run LLaMA-Factory CLI with that YAML.

if [ $# -lt 1 ]; then
  echo "Usage: $0 <examples/<...>.yaml | <custom args>>"
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT/LLaMA-Factory"

if [[ "$1" == *.yaml ]]; then
  llamafactory-cli train "$1"
else
  llamafactory-cli "$@"
fi
