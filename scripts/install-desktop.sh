#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
USER_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}"
APPLICATIONS_DIR="$USER_DATA_DIR/applications"
DESKTOP_FILE="$APPLICATIONS_DIR/joe.desktop"
ICON_SOURCE="$ROOT_DIR/src/joe/resources/joe.png"
ICON_DIR="$USER_DATA_DIR/icons/hicolor/64x64/apps"
ICON_FILE="$ICON_DIR/joe.png"
MIME_PACKAGES_DIR="$USER_DATA_DIR/mime/packages"
MIME_FILE="$MIME_PACKAGES_DIR/joe-audio-project.xml"
MIME_TYPE="application/x-joe-audio-project"
HOTKEY_PROPERTY="/commands/custom/<Primary><Shift>space"
HOTKEY_COMMAND="/usr/bin/bash \"$ROOT_DIR/run.sh\" --clipboard-audio"
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
  --uninstall  Remove lançadores, associação .joeaudio, ícone e atalho Xfce do Joe.
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

is_joe_mime_package() {
  [[ -f "$MIME_FILE" ]] && grep -Fqx '<!-- X-Joe-Managed=true -->' "$MIME_FILE"
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
  if command -v xfconf-query >/dev/null 2>&1; then
    current_hotkey="$(xfconf-query --channel xfce4-keyboard-shortcuts --property "$HOTKEY_PROPERTY" 2>/dev/null || true)"
    if [[ "$current_hotkey" == "$HOTKEY_COMMAND" ]]; then
      xfconf-query --channel xfce4-keyboard-shortcuts --property "$HOTKEY_PROPERTY" --reset >/dev/null 2>&1 || true
      echo "Atalho Ctrl+Shift+Espaço removido do Xfce."
    fi
  fi
  rm -f "$ICON_FILE"
  rmdir "$ICON_DIR" 2>/dev/null || true
  if is_joe_mime_package; then
    rm -f "$MIME_FILE"
    rmdir "$MIME_PACKAGES_DIR" 2>/dev/null || true
    if command -v update-mime-database >/dev/null 2>&1; then
      update-mime-database "$USER_DATA_DIR/mime" >/dev/null 2>&1 || true
    fi
    for mimeapps_file in "$CONFIG_DIR/mimeapps.list" "$APPLICATIONS_DIR/mimeapps.list"; do
      if [[ -f "$mimeapps_file" ]]; then
        MIME_TYPE="$MIME_TYPE" MIMEAPPS_FILE="$mimeapps_file" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MIMEAPPS_FILE"])
mime_type = os.environ["MIME_TYPE"]
original = path.read_text(encoding="utf-8")
lines = original.splitlines()
section = ""
updated = []
changed = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        section = stripped[1:-1]
    if section in {"Default Applications", "Added Associations", "Removed Associations"}:
        key, separator, value = line.partition("=")
        if separator and key.strip() == mime_type:
            applications = [
                item for item in value.strip().split(";")
                if item and item != "joe.desktop"
            ]
            changed = True
            if applications:
                updated.append(f"{key}={';'.join(applications)};")
            continue
    updated.append(line)

if changed:
    ending = "\n" if original.endswith("\n") else ""
    path.write_text("\n".join(updated) + ending, encoding="utf-8")
PY
      fi
    done
  fi
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
  fi
  echo "Lançadores, associação .joeaudio e ícone do Joe removidos (quando gerenciados pelo Joe)."
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
if [[ -e "$MIME_FILE" ]] && ! is_joe_mime_package; then
  echo "Já existe um pacote MIME não gerenciado em $MIME_FILE." >&2
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
    f"Exec=/usr/bin/bash {quote_exec_argument(str(run_script))} %f",
    "Icon=joe",
    "Terminal=false",
    "StartupNotify=true",
    "Categories=AudioVideo;Audio;Utility;",
    "Keywords=Joe;TTS;Texto;Áudio;Transcrição;",
    "MimeType=application/x-joe-audio-project;",
    "X-Joe-Managed=true",
    "",
])
destination.write_text(content, encoding="utf-8")
destination.chmod(0o644)
PY

mkdir -p "$MIME_PACKAGES_DIR"
cat > "$MIME_FILE" <<'MIME'
<?xml version="1.0" encoding="UTF-8"?>
<!-- X-Joe-Managed=true -->
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/x-joe-audio-project">
    <comment>Projeto Joe Audio Studio</comment>
    <icon name="joe"/>
    <glob pattern="*.joeaudio"/>
  </mime-type>
</mime-info>
MIME
chmod 644 "$MIME_FILE"

if command -v update-mime-database >/dev/null 2>&1; then
  update-mime-database "$USER_DATA_DIR/mime"
else
  echo "Aviso: update-mime-database não encontrado; a extensão pode não aparecer até atualizar os tipos MIME."
fi
if command -v xdg-mime >/dev/null 2>&1; then
  xdg-mime default joe.desktop "$MIME_TYPE"
else
  echo "Aviso: xdg-mime não encontrado; o Joe não foi definido como aplicativo padrão para .joeaudio."
fi

if [[ "$CREATE_DESKTOP" -eq 1 ]]; then
  mkdir -p "$DESKTOP_DIR"
  cp "$DESKTOP_FILE" "$DESKTOP_COPY"
  chmod 755 "$DESKTOP_COPY"
  echo "Atalho criado na área de trabalho: $DESKTOP_COPY"
  echo "Caso o Xfce peça confirmação, escolha 'Permitir iniciar'."
fi

if command -v xfconf-query >/dev/null 2>&1; then
  existing_hotkey="$(xfconf-query --channel xfce4-keyboard-shortcuts --property "$HOTKEY_PROPERTY" 2>/dev/null || true)"
  if [[ -z "$existing_hotkey" ]]; then
    if xfconf-query --channel xfce4-keyboard-shortcuts --property "$HOTKEY_PROPERTY" --create --type string --set "$HOTKEY_COMMAND" >/dev/null 2>&1; then
      echo "Atalho global Ctrl+Shift+Espaço registrado no Xfce."
    else
      echo "Aviso: não foi possível registrar o atalho no Xfce; o Joe tentará usar pynput."
    fi
  elif [[ "$existing_hotkey" == "$HOTKEY_COMMAND" ]]; then
    echo "Atalho global Ctrl+Shift+Espaço já está registrado no Xfce."
  else
    echo "Aviso: Ctrl+Shift+Espaço já pertence a outro comando no Xfce; não foi alterado."
    echo "O Joe tentará usar pynput como alternativa."
  fi
else
  echo "xfconf-query não encontrado; o Joe usará pynput como alternativa."
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
fi

echo "Joe instalado no menu de aplicativos:"
echo "  $DESKTOP_FILE"
echo "Associação de arquivos .joeaudio registrada para Joe Audio Studio."
echo "Ícone instalado em: $ICON_FILE"
echo "Procure por 'Joe Audio Studio' no menu."
