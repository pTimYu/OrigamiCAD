"""Layer stacking operations for solved hexagon models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, TypedDict

import numpy as np

from .metadata import CreaseKind

if TYPE_CHECKING:
    from ...core.cadder import Cadder


class LayerStackResult(TypedDict):
    """Geometry and diagnostics returned by :func:`stack_layers`."""

    model: Cadder
    num_layers: int
    layer_height: float
    valley_level: float
    expected_layer_height: Optional[float]
    max_interface_error: float


def analytical_layer_height(
    panel_distance: float,
    dihedral_angle: float,
    unit: Literal["rad", "deg"] = "deg",
) -> float:
    """Return ``d*sin(pi-theta)`` for one folded hexagon layer."""
    panel_distance = float(panel_distance)
    if not np.isfinite(panel_distance) or panel_distance <= 0.0:
        raise ValueError("panel_distance must be a finite positive value.")

    if unit == "deg":
        theta = float(np.deg2rad(dihedral_angle))
    elif unit == "rad":
        theta = float(dihedral_angle)
    else:
        raise ValueError("unit must be 'deg' or 'rad'.")

    if not np.isfinite(theta) or not (0.0 < theta < np.pi):
        raise ValueError("dihedral_angle must be between 0 and 180 degrees.")

    return float(panel_distance * np.sin(np.pi - theta))


def _triangle_surface_ids(
    model: Cadder,
    kind: CreaseKind,
) -> set[str]:
    if not model.hex_units:
        raise ValueError(
            "No hex-unit metadata found. Build the model with a hexagon "
            "layout function before stacking it."
        )

    surface_kinds: dict[str, CreaseKind] = {}
    for unit_data in model.hex_units:
        for triangle in unit_data.get("triangle_kinds", []):
            surface_id = triangle["surface"]
            triangle_kind = triangle["kind"]

            if surface_id not in model.surfaces:
                raise ValueError(
                    f"Hex-unit metadata references missing surface '{surface_id}'."
                )

            previous_kind = surface_kinds.get(surface_id)
            if previous_kind is not None and previous_kind != triangle_kind:
                raise ValueError(
                    f"Triangle '{surface_id}' is classified as both "
                    f"'{previous_kind}' and '{triangle_kind}'."
                )
            surface_kinds[surface_id] = triangle_kind

    surface_ids = {
        surface_id
        for surface_id, triangle_kind in surface_kinds.items()
        if triangle_kind == kind
    }
    if not surface_ids:
        raise ValueError(f"No {kind} triangle surfaces were found.")
    return surface_ids


def _common_surface_level(
    model: Cadder,
    surface_ids: set[str],
    tolerance: float,
    kind: CreaseKind,
) -> float:
    centroid_heights = []

    for surface_id in surface_ids:
        heights = np.array(
            [
                model.point_array(point_id)[2]
                for point_id in model._surface_vertices(surface_id)
            ],
            dtype=float,
        )
        if float(np.ptp(heights)) > tolerance:
            raise ValueError(
                f"{kind.capitalize()} panel '{surface_id}' is not horizontal "
                f"within tolerance {tolerance}."
            )
        centroid_heights.append(float(np.mean(heights)))

    if max(centroid_heights) - min(centroid_heights) > tolerance:
        raise ValueError(
            f"{kind.capitalize()} panels do not share one common height "
            f"within tolerance {tolerance}."
        )

    return float(np.mean(centroid_heights))


def layer_panel_levels(
    model: Cadder,
    tolerance: float = 1e-6,
) -> dict[CreaseKind, float]:
    """Return the common valley and mountain heights of a solved layer."""
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive value.")

    valley_surfaces = _triangle_surface_ids(model, "valley")
    mountain_surfaces = _triangle_surface_ids(model, "mountain")

    return {
        "valley": _common_surface_level(
            model,
            valley_surfaces,
            tolerance,
            "valley",
        ),
        "mountain": _common_surface_level(
            model,
            mountain_surfaces,
            tolerance,
            "mountain",
        ),
    }


def stack_layers(
    model: Cadder,
    num_layers: int = 4,
    expected_layer_height: Optional[float] = None,
    tolerance: float = 1e-6,
) -> LayerStackResult:
    """
    Build a static assembly from one solved hexagon layer.

    Even layers retain the solved orientation. Odd layers are reflected in
    height and exchange mountain/valley line labels. Object IDs are prefixed
    with ``layer_<index>::`` so coincident contact geometry remains separate.

    The returned assembly intentionally contains no kinematic constraints:
    stacking is a post-solve geometry operation.
    """
    if isinstance(num_layers, bool) or int(num_layers) != num_layers:
        raise ValueError("num_layers must be an integer.")
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive value.")

    panel_levels = layer_panel_levels(model, tolerance=tolerance)
    valley_level = panel_levels["valley"]
    layer_height = panel_levels["mountain"] - valley_level
    if layer_height <= tolerance:
        raise ValueError(
            "Mountain panels must lie above valley panels by more than "
            f"the stacking tolerance ({tolerance})."
        )

    if expected_layer_height is not None:
        expected_layer_height = float(expected_layer_height)
        if (
            not np.isfinite(expected_layer_height)
            or expected_layer_height <= 0.0
        ):
            raise ValueError(
                "expected_layer_height must be a finite positive value."
            )
        if abs(layer_height - expected_layer_height) > tolerance:
            raise ValueError(
                "The solved layer height does not match the expected height: "
                f"solved={layer_height:.9g}, "
                f"expected={expected_layer_height:.9g}."
            )

    assembly = type(model)(unit=model.unit)
    opposite_kind = {
        "mountain": "valley",
        "valley": "mountain",
    }

    for layer_index in range(num_layers):
        reverse = layer_index % 2 == 1
        prefix = f"layer_{layer_index}::"
        point_ids = {
            point_id: f"{prefix}{point_id}"
            for point_id in model.points
        }

        for point_id, point in model.points.items():
            relative_z = point.z - valley_level
            if reverse:
                z = (
                    valley_level
                    + (layer_index + 1) * layer_height
                    - relative_z
                )
            else:
                z = valley_level + layer_index * layer_height + relative_z

            assembly.add_point(
                point_ids[point_id],
                point.x,
                point.y,
                z,
            )

        for line_id in model.lines:
            start, end, kind = model._line_info(line_id)
            if reverse:
                kind = opposite_kind.get(kind, kind)
            assembly.lines[f"{prefix}{line_id}"] = {
                "start": point_ids[start],
                "end": point_ids[end],
                "kind": kind,
            }

        for surface_id in model.surfaces:
            assembly.surfaces[f"{prefix}{surface_id}"] = {
                "vertices": [
                    point_ids[point_id]
                    for point_id in model._surface_vertices(surface_id)
                ],
            }

    valley_surfaces = _triangle_surface_ids(model, "valley")
    mountain_surfaces = _triangle_surface_ids(model, "mountain")
    interface_errors = []

    for lower_index in range(num_layers - 1):
        contact_surfaces = (
            mountain_surfaces
            if lower_index % 2 == 0
            else valley_surfaces
        )
        lower_prefix = f"layer_{lower_index}::"
        upper_prefix = f"layer_{lower_index + 1}::"

        for surface_id in contact_surfaces:
            for point_id in model._surface_vertices(surface_id):
                interface_errors.append(
                    float(
                        np.linalg.norm(
                            assembly.point_array(f"{lower_prefix}{point_id}")
                            - assembly.point_array(f"{upper_prefix}{point_id}")
                        )
                    )
                )

    max_interface_error = max(interface_errors, default=0.0)
    if max_interface_error > tolerance:
        raise ValueError(
            "Adjacent layers do not stack within tolerance: "
            f"maximum interface error={max_interface_error:.9g}."
        )

    return {
        "model": assembly,
        "num_layers": num_layers,
        "layer_height": layer_height,
        "valley_level": valley_level,
        "expected_layer_height": expected_layer_height,
        "max_interface_error": max_interface_error,
    }


# Explicit pattern-specific alias for callers that prefer a descriptive name.
stack_simple_hexagon_layers = stack_layers
