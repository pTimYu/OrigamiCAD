"""Example: solve, project, and export a simple-hexagon origami structure.

Run from the project root:

    python -m examples.simple_hexagon

You can also run this file directly from an IDE.

Generated JSON/DXF/STEP files are written to ``examples/output/`` and ignored
by git through `.gitignore`.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import TwoDDrawer
from origamicad.patterns.hexagon_packaging import draw_hex_two_loops

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_two_loops(pattern)
    pattern.print_summary()
    pattern.draw(
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/2D_simple_hexagon.png"
    )

if __name__ == "__main__":
    main()
