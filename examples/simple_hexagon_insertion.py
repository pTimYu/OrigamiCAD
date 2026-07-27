"""Run the spatial multi-loop hexagon insertion simulation.

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
    draw_insertion_stack_3d,
    print_insertion_report,
    simulate_insertion,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SIDE_LENGTH = 15.0
# Exact contact-panel target; snap panels receive its supplementary angle.
REQUESTED_INNER_DIHEDRAL_DEG = 120.0
NUM_LOOPS = 3
NUM_LAYERS = 4
ASSIGNMENT_MODE = "panel_sequence"
REFERENCE_A2O_MASK = 0b111111
# This is unused by panel_sequence. In regular_masks mode, use None for the
# exhaustive search or [REFERENCE_A2O_MASK] for the sixfold paper state.
COMBINATION_MASKS = None


def main() -> None:
    result = simulate_insertion(
        inner_dihedral_deg=REQUESTED_INNER_DIHEDRAL_DEG,
        num_loops=NUM_LOOPS,
        num_layers=NUM_LAYERS,
        side_length=SIDE_LENGTH,
        assignment_mode=ASSIGNMENT_MODE,
        combination_masks=COMBINATION_MASKS,
        verbose=False,
    )
    print("")
    print_insertion_report(result)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # step_path = OUTPUT_DIR / "simple_hexagon_insertion_4layer_abaqus.step"
    # result["assembly"].save_step(
    #     str(step_path),
    #     thickness=0.0,
    #     separate_layer_parts=True,
    # )
    # print(f"Saved four-part Abaqus STEP:   {step_path}")

    drawing_path = OUTPUT_DIR / "simple_hexagon_insertion.png"
    draw_insertion_simulation(
        result,
        # save_path=drawing_path,
        show=False,
    )
    # print(f"Saved drawing:                 {drawing_path}")
    stack_drawing_path = (
        OUTPUT_DIR / "simple_hexagon_insertion_4layer_3d.png"
    )
    draw_insertion_stack_3d(
        result,
        # save_path=stack_drawing_path,
        show=True,
    )
    # print(f"Saved 3D stack drawing:        {stack_drawing_path}")


if __name__ == "__main__":
    main()
