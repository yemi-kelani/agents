"""Put `src/` on the path.

The application modules import each other flatly (`from settings import ...`)
and run with `src` as the working directory in CI, so the tests join that same
namespace rather than introducing a package layout the app does not use.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
