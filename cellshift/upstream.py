"""Import the vendored, unmodified upstream implementation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = (ROOT / 'vendor/scDisInFact/REVISION').read_text().strip()
sys.path.insert(0, str(ROOT / 'vendor/scDisInFact'))
from scDisInFact.model import create_scdisinfact_dataset, scdisinfact  # noqa: E402
