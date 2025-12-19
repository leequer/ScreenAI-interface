#!/bin/bash
set -euo pipefail

# Usage: bash scripts/run_rl.sh <VLM-R1/run_scripts/*.sh>
# For direct torchrun or python entrypoints, pass them after a --.

if [ $# -lt 1 ]; then
  echo "Usage: $0 <VLM-R1/run_scripts/*.sh | -- <cmd...>>"
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT/VLM-R1"

if [ "$1" == "--" ]; then
  shift
  "$@"
else
  bash "$1"
fi
