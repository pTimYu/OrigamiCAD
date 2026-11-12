"""Run the snap-through insertion simulation on a hexagon loop pattern.

The pattern is folded to the cavity angle, the innermost unit chain is held
at that angle, and a uniform pressure squeezes the outer boundary until the
other panels snap through.

Run from the project root:

    python -m examples.simple_hexagon_insertion
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad.patterns.hexagon import (
    draw_insertion_simulation,
    print_insertion_report,
    simulate_insertion,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SIDE_LENGTH = 15.0
# Dihedral angle at which the held cavity panels stay for the whole simulation.
INNER_DIHEDRAL_DEG = 135.0
NUM_LOOPS = 2
NUM_LAYERS = 4


def main() -> None:
    result = simulate_insertion(
        inner_dihedral_deg=INNER_DIHEDRAL_DEG,
        num_loops=NUM_LOOPS,
        num_layers=NUM_LAYERS,
        side_length=SIDE_LENGTH,
        # Crease stiffness per unit length, and how stiff panels are relative
        # to creases. Pressures in the report scale with the crease stiffness.
        hinge_stiffness=1.0,
        panel_stiffness_ratio=1e3,
        verbose=False,
    )
    print("")
    print_insertion_report(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drawing_path = OUTPUT_DIR / "simple_hexagon_insertion_135.png"
    draw_insertion_simulation(result, save_path=drawing_path, show=True)
    print(f"Saved drawing:                    {drawing_path}")


if __name__ == "__main__":
    main()
