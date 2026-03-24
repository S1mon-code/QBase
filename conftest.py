"""Path configuration for QBase.

Automatically adds QBase and AlphaForge to sys.path.
- pytest discovers this automatically
- Strategy scripts: `import conftest` at the top
"""
import sys
from pathlib import Path

from config import get_alphaforge_path

_qbase_root = str(Path(__file__).parent)
_af_path = get_alphaforge_path()

for p in [_qbase_root, _af_path]:
    if p not in sys.path:
        sys.path.insert(0, p)
