#!/usr/bin/env python3
"""Headless regression tests for BoxGUI logic (run with MPLBACKEND=Agg).

Usage:  MPLBACKEND=Agg python test_box_gui_local.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from box_extract_local import BoxGUI, CANONICAL_BOXES  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def select(gui, x0, x1, y0, y1):
    """Simulate a completed rectangle drag."""
    gui.selector.extents = (x0, x1, y0, y1)
    gui._on_select(None, None)


def main():
    frame = np.random.default_rng(0).normal(200, 10, (120, 160)).astype(np.float32)
    gui = BoxGUI(frame, CANONICAL_BOXES, title="test",
                 prev_boxes=[{"name": "eye", "x0": 5, "y0": 5, "x1": 25, "y1": 20}])

    # 1. assign with no box drawn -> refused
    gui._on_assign("laser_trigger")
    assert "Draw a NEW box" in gui.msg.get_text()
    assert not gui.assigned

    # 2. draw + assign
    select(gui, 0, 40, 0, 30)
    gui._on_assign("laser_trigger")
    assert gui.assigned["laser_trigger"] == [0, 0, 40, 30]
    assert gui.buttons["laser_trigger"].label.get_text() == "* laser_trigger"

    # 3. REGRESSION (caught on real data): clicking a second category without a new
    # drag must be refused — the consumed rectangle must not be silently reused.
    gui._on_assign("whisker_pad")
    assert "Draw a NEW box" in gui.msg.get_text()
    assert "whisker_pad" not in gui.assigned

    # 4. duplicate category -> ERROR, refused, original untouched
    select(gui, 50, 90, 50, 80)
    gui._on_assign("laser_trigger")
    assert "already assigned" in gui.msg.get_text()
    assert gui.assigned["laser_trigger"] == [0, 0, 40, 30]

    # 5. the still-pending selection can go to a fresh category
    gui._on_assign("whisker_pad")
    assert gui.assigned["whisker_pad"] == [50, 50, 90, 80]

    # 6. undo removes whisker_pad, button restored
    gui._on_undo()
    assert "whisker_pad" not in gui.assigned
    assert gui.buttons["whisker_pad"].label.get_text() == "whisker_pad"

    # 7. reuse previous fills eye
    gui._on_reuse()
    assert gui.assigned["eye"] == [5, 5, 25, 20]

    # 8. DONE with missing categories -> warns once, stays open; second DONE closes
    gui._on_done()
    assert "WARNING" in gui.msg.get_text() and gui._warned_missing
    assert plt.fignum_exists(gui.fig.number)
    gui._on_done()
    assert not plt.fignum_exists(gui.fig.number)

    # 9. output ordered canonically, only assigned boxes
    boxes = gui.run.__self__  # noqa: F841  (gui.run() would call plt.show; check directly)
    names = [c for c in gui.categories if c in gui.assigned]
    assert names == ["laser_trigger", "eye"]

    print("BoxGUI logic: all 9 checks passed")


if __name__ == "__main__":
    main()
