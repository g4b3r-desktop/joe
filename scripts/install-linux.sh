#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=python3

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

# Instala a build CPU do PyTorch para evitar baixar runtimes CUDA desnecessários.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[asr-fallback]"

echo
echo "Instalação concluída. Execute: ./run.sh"
