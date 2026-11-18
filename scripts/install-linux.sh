#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=python3
TORCH_MODE=cpu
INSTALL_LAUNCHER=1
CREATE_DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    cpu|cuda) TORCH_MODE="$arg" ;;
    --no-launcher) INSTALL_LAUNCHER=0 ;;
    --desktop) CREATE_DESKTOP=1 ;;
    --help|-h)
      echo "Uso: bash scripts/install-linux.sh [cpu|cuda] [--desktop] [--no-launcher]"
      echo "  cpu              Instala PyTorch CPU (padrão)."
      echo "  cuda             Instala PyTorch CUDA 13.2."
      echo "  --desktop        Também cria joe.desktop na área de trabalho."
      echo "  --no-launcher    Não registra o Joe no menu de aplicativos."
      exit 0
      ;;
    *)
      echo "Opção desconhecida: $arg" >&2
      echo "Use: bash scripts/install-linux.sh [cpu|cuda] [--desktop] [--no-launcher]"
      exit 2
      ;;
  esac
done

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

if [[ "$INSTALL_LAUNCHER" -eq 1 ]]; then
  launcher_args=()
  if [[ "$CREATE_DESKTOP" -eq 1 ]]; then
    launcher_args+=(--desktop)
  fi
  bash "$ROOT_DIR/scripts/install-desktop.sh" "${launcher_args[@]}"
fi

echo
echo "Instalação concluída. Abra 'Joe Audio Studio' pelo menu de aplicativos."
echo "Para iniciar pelo terminal: bash run.sh"
