from __future__ import annotations

import os
import sys


# Ensure repository root is on sys.path for imports like `import engine`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
