#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")/.." && pwd -P)"
APPLICATIONS_DIR="\${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APPLICATIONS_DIR/joe.desktop"
MODE="install"
CREATE_DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    --desktop) CREATE_DESKTOP=1 ;;
    --uninstall) MODE="uninstall" ;;
    --help|-h)
      cat <<'EOF'
Uso: bash scripts/install-desktop.sh [--desktop] [--uninstall]

Instala o lançador Joe no menu de aplicativos do usuário.
  --desktop    Também cria joe.desktop na área de trabalho.
  --uninstall  Remove os lançadores instalados pelo Joe.
EOF
      exit 0
      ;;
    *)
      echo "Opção desconhecida: $arg" >&2
      exit 2
      ;;
  esac
done

find_desktop_dir() {
  local candidate="\${XDG_DESKTOP_DIR:-}"
  if [[ -z "$candidate" ]] && command -v xdg-user-dir >/dev/null 2>&1; then
    candidate="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$HOME/Desktop"
  fi
  printf '%s\n' "$candidate"
}

DESKTOP_DIR="$(find_desktop_dir)"
DESKTOP_COPY="$DESKTOP_DIR/joe.desktop"

if [[ "$MODE" == "uninstall" ]]; then
  rm -f "$DESKTOP_FILE"
  if [[ -f "$DESKTOP_COPY" ]] && grep -q '^X-Joe-Managed=true$' "$DESKTOP_COPY"; then
    rm -f "$DESKTOP_COPY"
  fi
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
  fi
  echo "Lançador do Joe removido do menu de aplicativos."
  echo "Os arquivos do projeto e os modelos não foram removidos."
  exit 0
fi

if [[ ! -f "$ROOT_DIR/run.sh" ]]; then
  echo "Não encontrei run.sh em $ROOT_DIR." >&2
  exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Ambiente virtual do Joe não encontrado."
  echo "Execute primeiro: bash scripts/install-linux.sh"
  exit 1
fi

mkdir -p "$APPLICATIONS_DIR"

ROOT_DIR="$ROOT_DIR" DESKTOP_FILE="$DESKTOP_FILE" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"]).resolve()
destination = Path(os.environ["DESKTOP_FILE"])
run_script = root / "run.sh"

def quote_exec_argument(value: str) -> str:
    # Desktop Entry Exec quoting: escape reserved characters inside double quotes.
    escaped = value.replace("\\", "\\\\")
    for char in ('"', "\`", "$"):
        escaped = escaped.replace(char, "\\" + char)
    escaped = escaped.replace("%", "%%")
    return f'"{escaped}"'

content = "\n".join([
    "[Desktop Entry]",
    "Version=1.0",
    "Type=Application",
    "Name=Joe Audio Studio",
    "GenericName=Conversor de texto e áudio",
    "Comment=Converta texto em fala e transcreva áudio",
    f"Exec=/usr/bin/bash {quote_exec_argument(str(run_script))}",
    "Icon=audio-x-generic",
    "Terminal=false",
    "StartupNotify=true",
    "Categories=AudioVideo;Audio;Utility;",
    "Keywords=Joe;TTS;Texto;Áudio;Transcrição;",
    "X-Joe-Managed=true",
    "",
])
destination.write_text(content, encoding="utf-8")
destination.chmod(0o644)
PY

if [[ "$CREATE_DESKTOP" -eq 1 ]]; then
  mkdir -p "$DESKTOP_DIR"
  cp "$DESKTOP_FILE" "$DESKTOP_COPY"
  chmod 755 "$DESKTOP_COPY"
  echo "Atalho criado na área de trabalho: $DESKTOP_COPY"
  echo "Caso o Xfce peça confirmação, escolha 'Permitir iniciar'."
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
fi

echo "Joe instalado no menu de aplicativos:"
echo "  $DESKTOP_FILE"
echo "Procure por 'Joe Audio Studio' no menu."
