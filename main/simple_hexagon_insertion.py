"""Run the spatial two-loop hexagon insertion simulation.

Run from the project root:

    python -m main.simple_hexagon_insertion
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad.patterns.hexagon import (
    draw_insertion_simulation,
    draw_insertion_stack_3d,
    print_insertion_report,
    simulate_insertion,
)


OUTPUT_DIR = PROJECT_ROOT / "examples" / "output"
SIDE_LENGTH = 15.0
REQUESTED_INNER_DIHEDRAL_DEG = 150.0
NUM_LAYERS = 4
REFERENCE_A2O_MASK = 0b111111


def main() -> None:
    result = simulate_insertion(
        inner_dihedral_deg=REQUESTED_INNER_DIHEDRAL_DEG,
        num_layers=NUM_LAYERS,
        side_length=SIDE_LENGTH,
        combination_masks=[REFERENCE_A2O_MASK],
        verbose=True,
    )
    print("")
    print_insertion_report(result)
    OUTPUT_DIR.mkdir(exist_ok=True)
    drawing_path = OUTPUT_DIR / "simple_hexagon_insertion.png"
    draw_insertion_simulation(
        result,
        save_path=drawing_path,
        show=False,
    )
    print(f"Saved drawing:                 {drawing_path}")
    stack_drawing_path = (
        OUTPUT_DIR / "simple_hexagon_insertion_4layer_3d.png"
    )
    draw_insertion_stack_3d(
        result,
        save_path=stack_drawing_path,
        show=True,
    )
    print(f"Saved 3D stack drawing:        {stack_drawing_path}")


if __name__ == "__main__":
    main()
