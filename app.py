"""HuggingFace Spaces entry point for the LoanGuard Streamlit dashboard.

HF Spaces (Streamlit SDK) discovers and runs the file pointed to by
`app_file` in the README YAML frontmatter. This wrapper loads the
actual dashboard from src/dashboard/app.py so the project layout
stays clean.

Local users can still run:    streamlit run src/dashboard/app.py
HF Spaces will run:           streamlit run app.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Make `src` importable when this file runs from the repo root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Execute the actual dashboard module in this process so Streamlit
# sees its widgets / page config at the top level.
runpy.run_path(str(ROOT / "src" / "dashboard" / "app.py"), run_name="__main__")
