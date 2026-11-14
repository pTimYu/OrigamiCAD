"""Generate and draw the two-loop hexagon crease pattern.

Run from the project root:

    python -m examples.simple_hexagon_example_2d

You can also run this file directly from an IDE.
"""

from pathlib import Path

from origamicad import TwoDDrawer
from origamicad.patterns.hexagon import draw_hex_loops

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_loops(pattern, n=2)
    pattern.print_summary()
    pattern.draw(
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/2D_simple_hexagon.png"
    )

if __name__ == "__main__":
    main()
