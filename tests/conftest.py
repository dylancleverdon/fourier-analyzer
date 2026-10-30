import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FA_HOME", tempfile.mkdtemp(prefix="fa-test-"))
os.environ.setdefault("FA_FORCE_DSP", "1")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_engine_correction():
    """Each test starts with an uncorrected engine (corrections saved by one
    test must not change another's results)."""
    from fourier_analyzer.match.engine_correction import reset_saved
    reset_saved()
    yield
