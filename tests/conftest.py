import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FA_HOME", tempfile.mkdtemp(prefix="fa-test-"))
os.environ.setdefault("FA_FORCE_DSP", "1")
