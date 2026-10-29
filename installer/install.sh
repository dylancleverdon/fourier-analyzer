#!/bin/bash
# Fourier Analyzer installer for macOS (Apple silicon or Intel).
#
#   curl -fsSL https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/install.sh | bash
#
# Everything goes into your user account (no sudo):
#   ~/Library/Application Support/FourierAnalyzer/{app,venv,models,data}
#   /Applications/Fourier Analyzer.app   (or ~/Applications if /Applications is not writable)
#
# Environment overrides (optional):
#   FA_CHANNEL=branch:<name>   install from a branch instead of the latest release
#   FA_SOURCE=/path/to/checkout install from a local folder (development)
#   FA_SKIP_MODELS=1           do not pre-download the separation models
set -euo pipefail
# the whole script is wrapped in { } so `curl | bash` reads it completely before running
{

REPO="dylancleverdon/fourier-analyzer"
APP_NAME="Fourier Analyzer"
HOME_DIR="$HOME/Library/Application Support/FourierAnalyzer"
PY_VERSION="3.12"
CHANNEL="${FA_CHANNEL:-release}"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
step() { printf "\n\033[1;35m==>\033[0m \033[1m%s\033[0m\n" "$*"; }
warn() { printf "\033[33m%s\033[0m\n" "$*"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$*"; exit 1; }

[ "$(uname -s)" = "Darwin" ] || warn "This installer targets macOS; continuing anyway."
ARCH="$(uname -m)"
bold "Installing $APP_NAME ($ARCH, macOS $(sw_vers -productVersion 2>/dev/null || echo '?'))"
mkdir -p "$HOME_DIR"
LOG="$HOME_DIR/install.log"
exec > >(tee -a "$LOG") 2>&1
echo "---- install $(date) ----"

# ---------------------------------------------------------------- 1. uv + python
step "1/7  Python toolchain (uv)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed"
uv --version

# ---------------------------------------------------------------- 2. app code
step "2/7  Downloading the app ($CHANNEL)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if [ -n "${FA_SOURCE:-}" ]; then
  echo "using local source $FA_SOURCE"
  mkdir -p "$TMP/src"
  (cd "$FA_SOURCE" && tar --exclude .git -cf - .) | (cd "$TMP/src" && tar -xf -)
  VERSION="$(cat "$TMP/src/VERSION")"
else
  if [ "${CHANNEL#branch:}" != "$CHANNEL" ]; then
    BR="${CHANNEL#branch:}"
    URL="https://codeload.github.com/$REPO/tar.gz/refs/heads/$BR"
    VERSION=""
  else
    REL="$(curl -fsSL -H 'Accept: application/vnd.github+json' "https://api.github.com/repos/$REPO/releases/latest" || true)"
    VERSION="$(printf '%s' "$REL" | /usr/bin/python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("tag_name","").lstrip("v"))
except Exception: print("")' 2>/dev/null || true)"
    if [ -n "$VERSION" ]; then
      URL="https://github.com/$REPO/releases/download/v$VERSION/fourier-analyzer.tar.gz"
    else
      warn "no GitHub release found yet; installing the default branch"
      URL="https://codeload.github.com/$REPO/tar.gz/HEAD"
    fi
  fi
  echo "downloading $URL"
  curl -fsSL "$URL" -o "$TMP/app.tar.gz" || die "download failed: $URL"
  mkdir -p "$TMP/src"
  tar -xzf "$TMP/app.tar.gz" -C "$TMP/src"
  # GitHub source tarballs have one top-level folder
  if [ ! -d "$TMP/src/fourier_analyzer" ]; then
    INNER="$(find "$TMP/src" -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$INNER" "$TMP/src2" && rm -rf "$TMP/src" && mv "$TMP/src2" "$TMP/src"
  fi
  [ -n "$VERSION" ] && echo "$VERSION" > "$TMP/src/VERSION"
  VERSION="$(cat "$TMP/src/VERSION")"
fi
[ -d "$TMP/src/fourier_analyzer" ] || die "downloaded archive does not contain the app"
rm -rf "$HOME_DIR/app.old"
[ -d "$HOME_DIR/app" ] && mv "$HOME_DIR/app" "$HOME_DIR/app.old"
mv "$TMP/src" "$HOME_DIR/app"
rm -rf "$HOME_DIR/app.old"
echo "app version $VERSION"

# ---------------------------------------------------------------- 3. venv + deps
step "3/7  Python $PY_VERSION environment + dependencies (this takes a few minutes the first time)"
if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
  uv venv --seed --python "$PY_VERSION" "$HOME_DIR/venv"
fi
uv pip install --python "$HOME_DIR/venv/bin/python" -r "$HOME_DIR/app/requirements.txt"
uv pip install --python "$HOME_DIR/venv/bin/python" imageio-ffmpeg
# verify the separation stack imports (catches missing transitive deps early)
"$HOME_DIR/venv/bin/python" -c "import onnxruntime, audioread; from audio_separator.separator import Separator; print('separation stack OK')" \
  || { warn "separation stack incomplete; retrying install"; uv pip install --python "$HOME_DIR/venv/bin/python" --reinstall -r "$HOME_DIR/app/requirements.txt"; }
FFMPEG="$("$HOME_DIR/venv/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG" "$HOME_DIR/venv/bin/ffmpeg"
echo "ffmpeg: $FFMPEG"

# ---------------------------------------------------------------- 4. launcher + .app
step "4/7  Creating the $APP_NAME app"
cp "$HOME_DIR/app/installer/launcher.sh" "$HOME_DIR/fourier-analyzer"
chmod +x "$HOME_DIR/fourier-analyzer"
APPS="/Applications"
if ! [ -w "$APPS" ]; then APPS="$HOME/Applications"; mkdir -p "$APPS"; fi
BUNDLE="$APPS/$APP_NAME.app"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
cat > "$BUNDLE/Contents/MacOS/$APP_NAME" <<EOS
#!/bin/bash
exec "$HOME_DIR/fourier-analyzer" "\$@"
EOS
chmod +x "$BUNDLE/Contents/MacOS/$APP_NAME"
cat > "$BUNDLE/Contents/Info.plist" <<EOS
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>com.dylancleverdon.fourieranalyzer</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOS
# app icon (generated locally, no download)
if "$HOME_DIR/venv/bin/python" "$HOME_DIR/app/installer/make_icon.py" "$TMP/AppIcon.iconset" 2>/dev/null \
   && command -v iconutil >/dev/null 2>&1; then
  iconutil -c icns "$TMP/AppIcon.iconset" -o "$BUNDLE/Contents/Resources/AppIcon.icns" || true
fi
# unsigned app: make sure nothing is quarantined, then ad-hoc sign so Gatekeeper is happy
xattr -dr com.apple.quarantine "$BUNDLE" 2>/dev/null || true
xattr -dr com.apple.quarantine "$HOME_DIR" 2>/dev/null || true
codesign --force --deep --sign - "$BUNDLE" 2>/dev/null || true
echo "created $BUNDLE"

# ---------------------------------------------------------------- 5. models
step "5/7  Downloading separation models (Demucs + DrumSep, ~1 GB)"
if [ -z "${FA_SKIP_MODELS:-}" ]; then
  (cd "$HOME_DIR/app" && "$HOME_DIR/venv/bin/python" -m fourier_analyzer models) || warn "model download failed; the app will retry on first use"
fi

# ---------------------------------------------------------------- 6. Serum detection
step "6/7  Looking for Serum 2"
(cd "$HOME_DIR/app" && "$HOME_DIR/venv/bin/python" - <<'EOP'
from fourier_analyzer import config
print("Serum 2 plugin :", config.serum_plugin_path() or "NOT FOUND (verification in real Serum disabled)")
print("Serum tables   :", config.serum_tables_dir() or "NOT FOUND (set it in Settings)")
print("Serum presets  :", config.serum_presets_dir() or "NOT FOUND")
config.save_settings(update_channel=config.settings().get("update_channel", "release"),
                     auto_update=config.settings().get("auto_update", True))
EOP
) || true
if [ "${CHANNEL#branch:}" != "$CHANNEL" ]; then
  "$HOME_DIR/venv/bin/python" -c "
import sys; sys.path.insert(0, '$HOME_DIR/app')
from fourier_analyzer import config; config.save_settings(update_channel='$CHANNEL')"
fi

# ---------------------------------------------------------------- 7. done
step "7/7  Done"
bold "$APP_NAME $VERSION installed."
echo "  App:      $BUNDLE"
echo "  Exports:  ~/Music/Fourier Analyzer"
echo "  Presets:  Serum 2 browser -> User -> Fourier Analyzer"
echo "  Updates:  automatic on launch (Settings to change)"
echo "  Log:      $LOG"
if [ -z "${FA_NO_LAUNCH:-}" ]; then
  open "$BUNDLE" || true
fi
}
