#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Ambiente virtual não encontrado. Execute primeiro: ./scripts/install-linux.sh"
  exit 1
fi

exec .venv/bin/python -m joe "$@"
