#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=python3
TORCH_MODE=cpu
if [[ $# -ge 1 ]]; then
  TORCH_MODE="$1"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11+ não encontrado. Instale python3, python3-venv e tente novamente."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"Joe requer Python 3.11, 3.12 ou 3.13. Encontrado: {sys.version.split()[0]}"
    )
PY

if ! dpkg -s libsndfile1 >/dev/null 2>&1; then
  echo "Aviso: libsndfile1 não parece instalado. Em Debian/Ubuntu: sudo apt install libsndfile1"
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel
# Ensure switching between CPU and CUDA modes replaces the previous wheel.
python -m pip uninstall -y torch || true

case "$TORCH_MODE" in
  cpu)
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ;;
  cuda)
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
    ;;
  *)
    echo "Modo desconhecido: $TORCH_MODE. Use: cpu ou cuda"
    exit 2
    ;;
esac

python -m pip install -e ".[asr-fallback]"

echo
echo "Instalação concluída. Execute: bash run.sh"
