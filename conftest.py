"""Configure pytest to use this worktree's own source tree.

The jig package is installed in editable mode from the main worktree, but this
cohort branch carries implementation changes on top of that base. Inserting the
worktree's src/ at the front of sys.path ensures tests exercise the branch's
runner, pipeline, and adapter code rather than the installed package copy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
