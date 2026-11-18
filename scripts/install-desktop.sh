#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
USER_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
APPLICATIONS_DIR="$USER_DATA_DIR/applications"
DESKTOP_FILE="$APPLICATIONS_DIR/joe.desktop"
ICON_SOURCE="$ROOT_DIR/src/joe/resources/joe.png"
ICON_DIR="$USER_DATA_DIR/icons/hicolor/64x64/apps"
ICON_FILE="$ICON_DIR/joe.png"
MODE="install"
CREATE_DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    --desktop) CREATE_DESKTOP=1 ;;
    --uninstall) MODE="uninstall" ;;
    --help|-h)
      cat <<'HELP'
Uso: bash scripts/install-desktop.sh [--desktop] [--uninstall]

Instala o Joe no menu de aplicativos do usuário.
  --desktop    Também cria joe.desktop na área de trabalho.
  --uninstall  Remove os lançadores e o ícone instalados pelo Joe.
HELP
      exit 0
      ;;
    *)
      echo "Opção desconhecida: $arg" >&2
      exit 2
      ;;
  esac
done

find_desktop_dir() {
  local candidate="${XDG_DESKTOP_DIR:-}"
  if [[ -z "$candidate" ]] && command -v xdg-user-dir >/dev/null 2>&1; then
    candidate="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$HOME/Desktop"
  fi
  printf '%s\n' "$candidate"
}

is_joe_launcher() {
  [[ -f "$1" ]] && grep -Fqx 'X-Joe-Managed=true' "$1"
}

DESKTOP_DIR="$(find_desktop_dir)"
DESKTOP_COPY="$DESKTOP_DIR/joe.desktop"

if [[ "$MODE" == "uninstall" ]]; then
  if is_joe_launcher "$DESKTOP_FILE"; then
    rm -f "$DESKTOP_FILE"
  fi
  if is_joe_launcher "$DESKTOP_COPY"; then
    rm -f "$DESKTOP_COPY"
  fi
  rm -f "$ICON_FILE"
  rmdir "$ICON_DIR" 2>/dev/null || true
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
  fi
  echo "Lançadores e ícone do Joe removidos (quando gerenciados pelo Joe)."
  echo "O programa, as configurações e os modelos não foram removidos."
  exit 0
fi

if [[ ! -f "$ROOT_DIR/run.sh" ]]; then
  echo "Não encontrei run.sh em $ROOT_DIR." >&2
  exit 1
fi
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Ambiente virtual do Joe não encontrado. Execute primeiro: bash scripts/install-linux.sh" >&2
  exit 1
fi
if [[ ! -f "$ICON_SOURCE" ]]; then
  echo "Ícone do Joe não encontrado: $ICON_SOURCE" >&2
  exit 1
fi

if [[ -e "$DESKTOP_FILE" ]] && ! is_joe_launcher "$DESKTOP_FILE"; then
  echo "Já existe um lançador não gerenciado em $DESKTOP_FILE." >&2
  echo "Renomeie-o ou remova-o manualmente antes de instalar o Joe." >&2
  exit 1
fi
if [[ "$CREATE_DESKTOP" -eq 1 && -e "$DESKTOP_COPY" ]] && ! is_joe_launcher "$DESKTOP_COPY"; then
  echo "Já existe um arquivo não gerenciado em $DESKTOP_COPY." >&2
  echo "Renomeie-o ou remova-o manualmente antes de criar o atalho." >&2
  exit 1
fi

mkdir -p "$APPLICATIONS_DIR" "$ICON_DIR"
install -m 644 "$ICON_SOURCE" "$ICON_FILE"

ROOT_DIR="$ROOT_DIR" DESKTOP_FILE="$DESKTOP_FILE" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"]).resolve()
destination = Path(os.environ["DESKTOP_FILE"])
run_script = root / "run.sh"

def quote_exec_argument(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for char in ('"', chr(96), chr(36), chr(37)):
        escaped = escaped.replace(char, "\\" + char)
    return f'"{escaped}"'

content = "\n".join([
    "[Desktop Entry]",
    "Version=1.0",
    "Type=Application",
    "Name=Joe Audio Studio",
    "GenericName=Conversor de texto e áudio",
    "Comment=Converta texto em fala e transcreva áudio",
    f"Exec=/usr/bin/bash {quote_exec_argument(str(run_script))}",
    "Icon=joe",
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
echo "Ícone instalado em: $ICON_FILE"
echo "Procure por 'Joe Audio Studio' no menu."
