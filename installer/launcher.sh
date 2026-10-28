#!/bin/bash
# Started by "/Applications/Fourier Analyzer.app". Copied to
# ~/Library/Application Support/FourierAnalyzer/fourier-analyzer by the
# installer and refreshed by every update.
HOME_DIR="$HOME/Library/Application Support/FourierAnalyzer"
export PATH="$HOME_DIR/venv/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
cd "$HOME_DIR/app" || exit 1
exec "$HOME_DIR/venv/bin/python" -m fourier_analyzer app "$@" >> "$HOME_DIR/launcher.log" 2>&1
