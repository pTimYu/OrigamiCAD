"""Generate and draw a 2D hexagon-packaging crease pattern.

Run from the project root:

    python -m examples.hexagon_example_2d

You can also run this file directly from an IDE.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import TwoDDrawer
from origamicad.patterns.hexagon import build_packaging

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    build_packaging(
        pattern,
        l=15,
        alpha=2,
        beta=2,
        gamma=15,
        delta=7,
        enable_left_open=True,
        enable_hole_punch_outer=5
    )
    pattern.print_summary()
    pattern.draw(
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/2D_hexagon.png"
    )
    # pattern.save_dxf(f"{OUTPUT_DIR}/2D_hexagon.dxf", profile="solidworks");

if __name__ == "__main__":
    main()