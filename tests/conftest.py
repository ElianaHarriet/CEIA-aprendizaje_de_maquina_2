import sys
from pathlib import Path

# Add project root to sys.path so `src` is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
