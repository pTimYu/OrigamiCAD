"""Solve and export a four-layer stack of the two-loop hexagon pattern.

Run from the project root:

    python -m examples.simple_hexagon_four_layer_stack

Odd-numbered layers reverse the mountain/valley assignment.  Their geometry is
the vertical reflection of the solved base layer, so every upper valley
triangle rests on a mountain triangle from the layer below.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.io.cad_export import save_cad
from origamicad.patterns.hexagon_packaging import draw_hex_two_loops


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SIDE_LENGTH = 15.0
TARGET_DIHEDRAL_DEG = 135.0
NUM_LAYERS = 4


def _triangle_surface_ids(pattern: TwoDDrawer, kind: str) -> set[str]:
    """Return unique triangle surface IDs with the requested crease kind."""
    return {
        triangle["surface"]
        for unit in pattern.hex_units
        for triangle in unit["triangle_kinds"]
        if triangle["kind"] == kind
    }


def _mean_surface_height(model: Cadder, surface_ids: set[str]) -> float:
    """Return the mean centroid height of a collection of surfaces."""
    heights = [
        np.mean(
            [
                model.point_array(point_id)[2]
                for point_id in model._surface_vertices(surface_id)
            ]
        )
        for surface_id in surface_ids
    ]
    if not heights:
        raise ValueError("At least one surface is required to measure a height.")
    return float(np.mean(heights))


def _add_layer_to_assembly(
    assembly: Cadder,
    layer: Cadder,
    layer_index: int,
) -> None:
    """Copy one static layer into an assembly using namespaced object IDs."""
    prefix = f"layer_{layer_index}::"
    point_ids = {
        point_id: f"{prefix}{point_id}"
        for point_id in layer.points
    }

    for point_id, point in layer.points.items():
        assembly.add_point(
            point_ids[point_id],
            point.x,
            point.y,
            point.z,
        )

    for line_id in layer.lines:
        start, end, kind = layer._line_info(line_id)
        assembly.lines[f"{prefix}{line_id}"] = {
            "start": point_ids[start],
            "end": point_ids[end],
            "kind": kind,
        }

    for surface_id in layer.surfaces:
        assembly.surfaces[f"{prefix}{surface_id}"] = {
            "vertices": [
                point_ids[point_id]
                for point_id in layer._surface_vertices(surface_id)
            ],
        }


def _interface_error(
    lower_model: Cadder,
    lower_pattern: TwoDDrawer,
    upper_model: Cadder,
    upper_pattern: TwoDDrawer,
) -> float:
    """Return the largest point mismatch at one mountain/valley interface."""
    lower_surfaces = _triangle_surface_ids(lower_pattern, "mountain")
    upper_surfaces = _triangle_surface_ids(upper_pattern, "valley")
    if lower_surfaces != upper_surfaces:
        raise ValueError(
            "Adjacent layers do not expose matching mountain/valley panels."
        )

    errors = []
    for surface_id in lower_surfaces:
        lower_vertices = lower_model._surface_vertices(surface_id)
        upper_vertices = upper_model._surface_vertices(surface_id)
        if lower_vertices != upper_vertices:
            raise ValueError(
                f"Panel '{surface_id}' has inconsistent vertex ordering."
            )

        errors.extend(
            np.linalg.norm(
                lower_model.point_array(lower_point)
                - upper_model.point_array(upper_point)
            )
            for lower_point, upper_point in zip(
                lower_vertices,
                upper_vertices,
            )
        )

    return float(max(errors, default=0.0))


def main() -> None:
    base_pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_two_loops(
        base_pattern,
        start_point=(0.0, 0.0),
        l=SIDE_LENGTH,
        reverse=False,
    )

    base_model = Cadder.from_drawer(base_pattern)
    base_model.solve_simple_hexagon_kinematics(
        final_dihedral=TARGET_DIHEDRAL_DEG,
        start_dihedral=175.0,
        steps=4,
        unit="deg",
        fixed_triangle_surface_id="tri_0_1",
        valley_z=0.0,
        strict_unique_edges=False,
        mountain_height=2.0,
        valley_height=0.0,
        max_nfev_per_step=8000,
        tol=1e-10,
        verbose=True,
    )

    # d is the perpendicular distance between the two parallel crease axes
    # across a quadrilateral panel.
    d = SIDE_LENGTH * np.sqrt(3.0) / 2.0
    theta = np.deg2rad(TARGET_DIHEDRAL_DEG)
    expected_layer_height = float(d * np.sin(np.pi - theta))

    valley_height = _mean_surface_height(
        base_model,
        _triangle_surface_ids(base_pattern, "valley"),
    )
    mountain_height = _mean_surface_height(
        base_model,
        _triangle_surface_ids(base_pattern, "mountain"),
    )
    solved_layer_height = mountain_height - valley_height

    height_tolerance = 1e-6 * max(1.0, SIDE_LENGTH)
    if abs(solved_layer_height - expected_layer_height) > height_tolerance:
        raise ValueError(
            "The solved layer height does not match d*sin(pi-theta): "
            f"solved={solved_layer_height:.9g}, "
            f"expected={expected_layer_height:.9g}."
        )

    assembly = Cadder(unit=base_model.unit)
    layers: list[tuple[TwoDDrawer, Cadder]] = []

    for layer_index in range(NUM_LAYERS):
        reverse = layer_index % 2 == 1
        layer_pattern = TwoDDrawer(unit=base_model.unit, point_tol=1e-6)
        draw_hex_two_loops(
            layer_pattern,
            start_point=(0.0, 0.0),
            l=SIDE_LENGTH,
            reverse=reverse,
        )
        layer_model = Cadder.from_drawer(layer_pattern)

        for point_id, base_point in base_model.points.items():
            layer_point = layer_model.points[point_id]
            layer_point.x = base_point.x
            layer_point.y = base_point.y

            base_z = base_point.z - valley_height
            if reverse:
                layer_point.z = (
                    (layer_index + 1) * solved_layer_height - base_z
                )
            else:
                layer_point.z = layer_index * solved_layer_height + base_z

        _add_layer_to_assembly(assembly, layer_model, layer_index)
        layers.append((layer_pattern, layer_model))

    max_interface_error = max(
        _interface_error(
            lower_model,
            lower_pattern,
            upper_model,
            upper_pattern,
        )
        for (
            lower_pattern,
            lower_model,
        ), (
            upper_pattern,
            upper_model,
        ) in zip(layers, layers[1:])
    )
    if max_interface_error > height_tolerance:
        raise ValueError(
            "Adjacent layers do not stack within tolerance: "
            f"maximum interface error={max_interface_error:.9g}."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "simple_hexagon_four_layer_stack.step"
    save_cad(assembly, output_path)

    print(f"Analytical layer height: {expected_layer_height:.6f} mm")
    print(f"Solved layer height:     {solved_layer_height:.6f} mm")
    print(f"Maximum interface error: {max_interface_error:.3e} mm")
    print(f"Saved four-layer stack:  {output_path}")

    assembly.draw(
        show_surfaces=True,
        figsize=(10, 10),
    )


if __name__ == "__main__":
    main()
