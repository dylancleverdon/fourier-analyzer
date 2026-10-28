#!/bin/bash
# Removes the app, its Python environment, models and projects.
# Exported files in ~/Music/Fourier Analyzer and installed Serum presets are kept.
set -e
rm -rf "/Applications/Fourier Analyzer.app" "$HOME/Applications/Fourier Analyzer.app"
rm -rf "$HOME/Library/Application Support/FourierAnalyzer"
echo "Fourier Analyzer removed."
