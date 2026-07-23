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
    build_packaging(pattern, l=15, alpha=3, beta=3, gamma=3, delta=3)
    pattern.print_summary()
    pattern.draw(
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/2D_hexagon.png"
    )

if __name__ == "__main__":
    main()
