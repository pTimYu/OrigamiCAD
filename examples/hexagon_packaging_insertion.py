"""Snap-through insertion of a rectangular package that is open on one side.

A wall around a cavity that is open on one side has no closed outer boundary
to press on. The simulation therefore completes it with the same lattice
continued over its 180-degree rotation, squeezes the closed ring, and returns
the half that was supplied. The cavity panels are held at the cavity angle.

Run from the project root:

    python -m examples.hexagon_packaging_insertion
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import (
    build_packaging,
    draw_insertion_simulation,
    print_insertion_report,
    simulate_insertion,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    build_packaging(
        pattern, l=15.0, alpha=1, beta=1, gamma=2, delta=2, enable_left_open=True,
    )
    model = Cadder.from_drawer(pattern)

    result = simulate_insertion(
        inner_dihedral_deg=135.0,
        num_layers=2,
        model=model,
        # The pressure the package is squeezed to. Without it the ramp
        # continues until no panel snaps while the pressure triples.
        pressure_max=0.6,
    )
    print("")
    print_insertion_report(result)
    print(f"Panels in the supplied half:      {len(result['model'].surfaces)}")
    print(f"Panels in the simulated ring:     {len(result['closed_model'].surfaces)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drawing_path = OUTPUT_DIR / "hexagon_packaging_insertion.png"
    draw_insertion_simulation(result, save_path=drawing_path, show=True)
    print(f"Saved drawing:                    {drawing_path}")


if __name__ == "__main__":
    main()
