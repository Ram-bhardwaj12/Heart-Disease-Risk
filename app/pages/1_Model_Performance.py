"""Forwarder to app/views/performance.py for backward compatibility with direct page runners."""

from pathlib import Path
import runpy

view_path = Path(__file__).resolve().parent.parent / "views" / "performance.py"
runpy.run_path(str(view_path), run_name="__main__")
