#!/bin/bash
# Double-click installer. If macOS says it cannot be opened because it is from
# an unidentified developer: right-click -> Open, or System Settings ->
# Privacy & Security -> "Open Anyway". Everything else is automatic.
cd "$(dirname "$0")"
if [ -f "./install.sh" ]; then
  bash ./install.sh
else
  curl -fsSL https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/install.sh | bash
fi
echo
read -n 1 -s -r -p "Press any key to close this window"
