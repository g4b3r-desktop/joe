#!/usr/bin/env bash
set -euo pipefail
CALLER_DIR="$PWD"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Preserve the caller's working directory when a relative .joeaudio path is
# opened by the desktop file manager.
if [[ "${1:-}" == *.joeaudio && "${1:-}" != /* ]]; then
  set -- "$CALLER_DIR/$1" "${@:2}"
fi

cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Ambiente virtual não encontrado. Execute primeiro: ./scripts/install-linux.sh"
  exit 1
fi

exec .venv/bin/python -m joe "$@"
