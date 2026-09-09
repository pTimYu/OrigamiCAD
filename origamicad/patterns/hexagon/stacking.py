"""Layer stacking and aligned-pattern operations for solved hexagon models."""

from __future__ import annotations

from collections.abc import Sequence
import copy
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import numpy as np

from .metadata import CreaseKind

if TYPE_CHECKING:
    from ...core.cadder import Cadder
    from ...core.two_d_drawer import TwoDDrawer


class LayerInterfaceResult(TypedDict):
    """Adhesive-contact diagnostics for two adjacent stacked layers."""

    lower_layer: int
    upper_layer: int
    contact_kind: CreaseKind
    num_lower_surfaces: int
    num_upper_surfaces: int
    num_matched_surfaces: int
    unmatched_upper_surface_ids: list[str]
    adhesive_area: float
    max_interface_error: float


class MixedLayerStackResult(TypedDict):
    """Geometry and diagnostics returned by :func:`stack_mixed_layers`."""

    model: Cadder
    num_layers: int
    layer_height: float
    layer_heights: list[float]
    valley_levels: list[float]
    interfaces: list[LayerInterfaceResult]
    max_interface_error: float


class LayerStackResult(TypedDict):
    """Geometry and diagnostics returned by :func:`stack_layers`."""

    model: Cadder
    num_layers: int
    layer_height: float
    valley_level: float
    max_interface_error: float


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive value.")
    return tolerance


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
    tolerance = _validate_tolerance(tolerance)
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


def _panel_model_from_pattern(pattern: TwoDDrawer) -> Cadder:
    """Return a solve-ready model containing only panel-connected geometry.

    Hole-punch polygons are cut-line metadata rather than kinematic panels.
    Removing their unconstrained points before solving makes large packaging
    patterns substantially faster. Hole contours are restored internally by
    :func:`stack_mixed_layers` after the reference solve.
    """
    from ...core.cadder import Cadder

    model = Cadder.from_drawer(pattern)
    panel_point_ids = {
        point_id
        for surface_id in model.surfaces
        for point_id in model._surface_vertices(surface_id)
    }

    model.points = {
        point_id: point
        for point_id, point in model.points.items()
        if point_id in panel_point_ids
    }
    model.lines = {
        line_id: line
        for line_id, line in model.lines.items()
        if line["start"] in panel_point_ids
        and line["end"] in panel_point_ids
    }
    model.surface_holes = {}
    return model


def _inherit_folded_geometry(
    subset_pattern: TwoDDrawer,
    reference_pattern: TwoDDrawer,
    reference_model: Cadder,
    tolerance: float = 1e-6,
) -> Cadder:
    """Create a folded subset model from an already solved reference pattern.

    Every panel vertex and surface in ``subset_pattern`` must also occur in
    ``reference_pattern``. Coordinates are copied from ``reference_model`` so
    common panels are exactly congruent and can serve as adhesive contacts.
    This is useful for cavity/no-cavity variants generated from one layout.
    """
    tolerance = _validate_tolerance(tolerance)
    if subset_pattern.unit != reference_pattern.unit:
        raise ValueError("The subset and reference patterns must use one unit.")
    if reference_model.unit != reference_pattern.unit:
        raise ValueError("The reference model and pattern must use one unit.")

    subset_model = _panel_model_from_pattern(subset_pattern)
    reference_surface_keys = {
        frozenset(reference_model._surface_vertices(surface_id))
        for surface_id in reference_model.surfaces
    }
    point_mapping: dict[str, str] = {}

    for point_id, point in subset_model.points.items():
        source_point = subset_pattern.points[point_id]
        reference_point_id = reference_pattern.find_point_by_coordinate(
            source_point.x,
            source_point.y,
            tol=tolerance,
        )
        if (
            reference_point_id is None
            or reference_point_id not in reference_model.points
        ):
            raise ValueError(
                "The subset pattern has no reference-panel match for point "
                f"{point_id!r}."
            )

        point_mapping[point_id] = reference_point_id
        reference_point = reference_model.points[reference_point_id]
        point.x = reference_point.x
        point.y = reference_point.y
        point.z = reference_point.z

    for surface_id in subset_model.surfaces:
        mapped_vertices = frozenset(
            point_mapping[point_id]
            for point_id in subset_model._surface_vertices(surface_id)
        )
        if mapped_vertices not in reference_surface_keys:
            raise ValueError(
                "The subset pattern has no reference-panel match for surface "
                f"{surface_id!r}."
            )

    return subset_model


def _hole_host_quad(
    pattern: TwoDDrawer,
    hole: dict,
    tolerance: float,
) -> str:
    """Return the parallelogram whose centroid defines a hole punch."""
    hole_center = np.asarray(hole["center"], dtype=float)

    for surface_id, surface in pattern.surfaces.items():
        if len(surface.vertices) != 4:
            continue
        center = np.mean(
            [
                [pattern.points[point_id].x, pattern.points[point_id].y]
                for point_id in surface.vertices
            ],
            axis=0,
        )
        if float(np.linalg.norm(center - hole_center)) <= tolerance:
            return surface_id

    raise ValueError(f"Could not find the host panel for hole {hole['id']!r}.")


def _add_folded_hole_punches(
    pattern: TwoDDrawer,
    folded_model: Cadder,
    tolerance: float = 1e-6,
) -> int:
    """Rigidly map a pattern's hole-punch cut lines onto folded panels.

    Existing hole points and lines are updated, so the operation also works on
    a model created directly with ``Cadder.from_drawer``. The return value is
    the number of mapped hole punches.
    """
    tolerance = _validate_tolerance(tolerance)
    if pattern.unit != folded_model.unit:
        raise ValueError("The pattern and folded model must use one unit.")

    source_lines = {
        tuple(sorted((line.start, line.end))): (line_id, line)
        for line_id, line in pattern.lines.items()
    }

    for hole in pattern.hole_punches:
        host_surface_id = _hole_host_quad(pattern, hole, tolerance)
        if host_surface_id not in folded_model.surfaces:
            raise ValueError(
                f"Folded model is missing hole host surface {host_surface_id!r}."
            )
        host_point_ids = pattern.surfaces[host_surface_id].vertices
        if any(point_id not in folded_model.points for point_id in host_point_ids):
            raise ValueError(
                f"Folded model is missing vertices of {host_surface_id!r}."
            )

        planar_quad = np.asarray(
            [
                [pattern.points[point_id].x, pattern.points[point_id].y]
                for point_id in host_point_ids
            ],
            dtype=float,
        )
        folded_quad = np.asarray(
            [folded_model.point_array(point_id) for point_id in host_point_ids],
            dtype=float,
        )

        planar_basis = np.column_stack(
            (planar_quad[1] - planar_quad[0], planar_quad[3] - planar_quad[0])
        )
        if abs(float(np.linalg.det(planar_basis))) <= tolerance:
            raise ValueError(
                f"Hole host surface {host_surface_id!r} is degenerate."
            )
        folded_axis_a = folded_quad[1] - folded_quad[0]
        folded_axis_b = folded_quad[3] - folded_quad[0]
        hole_point_ids = [
            f"{hole['id']}_p{index}"
            for index in range(int(hole["segments"]))
        ]

        for point_id in hole_point_ids:
            if point_id not in pattern.points:
                raise ValueError(
                    f"Pattern is missing point {point_id!r} for hole "
                    f"{hole['id']!r}."
                )
            source_point = pattern.points[point_id]
            local_coordinates = np.linalg.solve(
                planar_basis,
                np.asarray([source_point.x, source_point.y]) - planar_quad[0],
            )
            folded_point = (
                folded_quad[0]
                + local_coordinates[0] * folded_axis_a
                + local_coordinates[1] * folded_axis_b
            )
            if point_id in folded_model.points:
                target_point = folded_model.points[point_id]
                target_point.x, target_point.y, target_point.z = map(
                    float,
                    folded_point,
                )
            else:
                folded_model.add_point(point_id, *folded_point)

        for index, point_id in enumerate(hole_point_ids):
            next_point_id = hole_point_ids[(index + 1) % len(hole_point_ids)]
            edge = tuple(sorted((point_id, next_point_id)))
            if edge not in source_lines:
                raise ValueError(
                    f"Pattern is missing the cut-line edge {edge!r}."
                )
            line_id, line = source_lines[edge]
            folded_model.lines[line_id] = {
                "start": line.start,
                "end": line.end,
                "kind": line.kind,
            }

        surface_holes = folded_model.surface_holes.setdefault(
            host_surface_id,
            [],
        )
        if hole_point_ids not in surface_holes:
            surface_holes.append(hole_point_ids)

    return len(pattern.hole_punches)


def _surface_coordinates(model: Cadder, surface_id: str) -> np.ndarray:
    return np.asarray(
        [
            model.point_array(point_id)
            for point_id in model._surface_vertices(surface_id)
        ],
        dtype=float,
    )


def _unordered_vertex_error(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        max(
            np.max(np.min(distances, axis=0)),
            np.max(np.min(distances, axis=1)),
        )
    )


def _interface_diagnostics(
    assembly: Cadder,
    layer_models: list[Cadder],
    lower_index: int,
    tolerance: float,
) -> LayerInterfaceResult:
    upper_index = lower_index + 1
    contact_kind: CreaseKind = (
        "mountain" if lower_index % 2 == 0 else "valley"
    )
    lower_surface_ids = _triangle_surface_ids(
        layer_models[lower_index],
        contact_kind,
    )
    upper_surface_ids = _triangle_surface_ids(
        layer_models[upper_index],
        contact_kind,
    )

    available_lower = set(lower_surface_ids)
    lower_coordinates = {
        surface_id: _surface_coordinates(
            assembly,
            f"layer_{lower_index}::{surface_id}",
        )
        for surface_id in lower_surface_ids
    }
    lower_ids = list(lower_coordinates)
    lower_bounds = np.array([
        (coordinates.min(axis=0), coordinates.max(axis=0))
        for coordinates in lower_coordinates.values()
    ])
    # The exact norm can underflow for extremely small separations. A loose
    # bound in that range keeps the existing distance/tolerance decision.
    bound_tolerance = max(np.nextafter(tolerance, np.inf), np.sqrt(np.finfo(float).tiny))
    unmatched_upper = []
    matched_errors = []
    adhesive_area = 0.0

    for upper_surface_id in sorted(upper_surface_ids):
        stacked_upper_id = f"layer_{upper_index}::{upper_surface_id}"
        upper_coordinates = _surface_coordinates(assembly, stacked_upper_id)
        best_lower_id = None
        best_error = float("inf")
        # A vertex match within tolerance requires both coordinate bounds to
        # agree within tolerance. Keep the original set order for tie breaking.
        upper_bounds = np.array((upper_coordinates.min(axis=0), upper_coordinates.max(axis=0)))
        candidate_ids = {
            lower_ids[index]
            for index in np.flatnonzero(np.all(
                np.abs(lower_bounds - upper_bounds) <= bound_tolerance,
                axis=(1, 2),
            ))
        }

        for lower_surface_id in available_lower:
            if lower_surface_id not in candidate_ids:
                continue
            error = _unordered_vertex_error(
                lower_coordinates[lower_surface_id],
                upper_coordinates,
            )
            if error < best_error:
                best_error = error
                best_lower_id = lower_surface_id

        if best_lower_id is None or best_error > tolerance:
            unmatched_upper.append(upper_surface_id)
            continue

        available_lower.remove(best_lower_id)
        matched_errors.append(best_error)
        adhesive_area += 0.5 * float(
            np.linalg.norm(
                np.cross(
                    upper_coordinates[1] - upper_coordinates[0],
                    upper_coordinates[2] - upper_coordinates[0],
                )
            )
        )

    return {
        "lower_layer": lower_index,
        "upper_layer": upper_index,
        "contact_kind": contact_kind,
        "num_lower_surfaces": len(lower_surface_ids),
        "num_upper_surfaces": len(upper_surface_ids),
        "num_matched_surfaces": len(matched_errors),
        "unmatched_upper_surface_ids": unmatched_upper,
        "adhesive_area": adhesive_area,
        "max_interface_error": max(matched_errors, default=0.0),
    }


def _prepare_layer_models(
    layers: list[Cadder | TwoDDrawer],
    *,
    reference_layer: int,
    reference_model: Optional[Cadder],
    solve_options: Optional[dict[str, Any]],
    include_hole_punches: bool,
    tolerance: float,
) -> list[Cadder]:
    """Turn either solved models or compatible 2D patterns into layer models."""
    from ...core.cadder import Cadder
    from ...core.two_d_drawer import TwoDDrawer

    if all(isinstance(layer, Cadder) for layer in layers):
        if reference_model is not None or solve_options is not None:
            raise ValueError(
                "reference_model and solve_options apply only when layers "
                "contains TwoDDrawer patterns."
            )
        return list(layers)

    if not all(isinstance(layer, TwoDDrawer) for layer in layers):
        raise ValueError(
            "layers must contain either only Cadder models or only "
            "TwoDDrawer patterns."
        )
    if (
        isinstance(reference_layer, bool)
        or int(reference_layer) != reference_layer
    ):
        raise ValueError("reference_layer must be an integer.")
    reference_layer = int(reference_layer)
    if not 0 <= reference_layer < len(layers):
        raise ValueError(
            f"reference_layer must be between 0 and {len(layers) - 1}."
        )

    patterns = list(layers)
    reference_pattern = patterns[reference_layer]
    if reference_model is None:
        from .kinematics import solve_kinematics

        prepared_reference = _panel_model_from_pattern(reference_pattern)
        solve_result = solve_kinematics(
            prepared_reference,
            **dict(solve_options or {}),
        )
        if not solve_result["report"].success:
            raise RuntimeError(solve_result["report"].message)
    else:
        if solve_options is not None:
            raise ValueError(
                "solve_options cannot be supplied with reference_model."
            )
        prepared_reference = copy.deepcopy(reference_model)

    prepared_by_pattern_id: dict[int, Cadder] = {
        id(reference_pattern): prepared_reference,
    }
    for pattern in patterns:
        pattern_id = id(pattern)
        if pattern_id not in prepared_by_pattern_id:
            prepared_by_pattern_id[pattern_id] = _inherit_folded_geometry(
                pattern,
                reference_pattern,
                prepared_reference,
                tolerance=tolerance,
            )

    if include_hole_punches:
        unique_patterns = {id(pattern): pattern for pattern in patterns}
        for pattern_id, pattern in unique_patterns.items():
            _add_folded_hole_punches(
                pattern,
                prepared_by_pattern_id[pattern_id],
                tolerance=tolerance,
            )
    else:
        for model in prepared_by_pattern_id.values():
            model.surface_holes = {}

    return [prepared_by_pattern_id[id(pattern)] for pattern in patterns]


def stack_mixed_layers(
    layers: Sequence[Cadder] | Sequence[TwoDDrawer],
    tolerance: float = 1e-6,
    require_upper_support: bool = True,
    reference_layer: int = 0,
    reference_model: Optional[Cadder] = None,
    solve_options: Optional[dict[str, Any]] = None,
    include_hole_punches: bool = True,
    layer_counts: Optional[Sequence[int]] = None,
) -> MixedLayerStackResult:
    """Prepare and stack a bottom-to-top sequence of hexagon layers.

    Even layers retain their solved orientation. Odd layers are reflected in
    height and exchange mountain/valley line labels. Adjacent contact
    triangles are matched geometrically rather than by object ID, allowing a
    cavity layer to adhere to the corresponding subset of a filled layer.

    ``layers`` may contain solved ``Cadder`` models or ``TwoDDrawer`` patterns.
    Use ``layer_counts`` to repeat each entry without manually expanding it;
    for example, ``layers=[filled, cavity], layer_counts=[4, 6]``.
    For patterns, the pattern at ``reference_layer`` is solved using
    ``solve_options`` (or an existing ``reference_model``), compatible subset
    patterns inherit its exact folded geometry, and hole punches are restored
    automatically when ``include_hole_punches=True``. Thus a complete mixed
    stack can be created in one call.

    Every prepared model must use the same unit and have the same
    mountain-to-valley height. With ``require_upper_support=True`` (the
    default), every contact triangle on an upper layer must have a coincident
    triangle on the layer below.
    """
    source_layers = list(layers)
    if not source_layers:
        raise ValueError("layers must contain at least one model or pattern.")
    if layer_counts is None:
        raw_layers = source_layers
    else:
        counts = list(layer_counts)
        if len(counts) != len(source_layers):
            raise ValueError("layer_counts must have one value per layer entry.")
        if any(
            isinstance(count, bool)
            or int(count) != count
            or int(count) < 1
            for count in counts
        ):
            raise ValueError("Every layer_counts value must be a positive integer.")
        raw_layers = [
            layer
            for layer, count in zip(source_layers, counts)
            for _ in range(int(count))
        ]
    if not isinstance(require_upper_support, bool):
        raise ValueError("require_upper_support must be a boolean.")
    if not isinstance(include_hole_punches, bool):
        raise ValueError("include_hole_punches must be a boolean.")
    tolerance = _validate_tolerance(tolerance)
    models = _prepare_layer_models(
        raw_layers,
        reference_layer=reference_layer,
        reference_model=reference_model,
        solve_options=solve_options,
        include_hole_punches=include_hole_punches,
        tolerance=tolerance,
    )

    model_unit = models[0].unit
    if any(model.unit != model_unit for model in models[1:]):
        raise ValueError("All layer models must use the same unit.")

    levels_by_model = {}
    for model in models:
        if id(model) not in levels_by_model:
            levels_by_model[id(model)] = layer_panel_levels(model, tolerance=tolerance)
    panel_levels = [levels_by_model[id(model)] for model in models]
    valley_levels = [levels["valley"] for levels in panel_levels]
    layer_heights = [
        levels["mountain"] - levels["valley"]
        for levels in panel_levels
    ]
    if any(height <= tolerance for height in layer_heights):
        raise ValueError(
            "Mountain panels must lie above valley panels by more than "
            f"the stacking tolerance ({tolerance})."
        )

    layer_height = float(np.mean(layer_heights))
    if max(abs(height - layer_height) for height in layer_heights) > tolerance:
        raise ValueError(
            "All layer models must have the same mountain-to-valley height "
            f"within tolerance {tolerance}; got {layer_heights}."
        )

    assembly = type(models[0])(unit=model_unit)
    opposite_kind = {
        "mountain": "valley",
        "valley": "mountain",
    }
    base_valley_level = valley_levels[0]

    for layer_index, model in enumerate(models):
        reverse = layer_index % 2 == 1
        prefix = f"layer_{layer_index}::"
        point_ids = {
            point_id: f"{prefix}{point_id}"
            for point_id in model.points
        }

        for point_id, point in model.points.items():
            relative_z = point.z - valley_levels[layer_index]
            if reverse:
                z = (
                    base_valley_level
                    + (layer_index + 1) * layer_height
                    - relative_z
                )
            else:
                z = (
                    base_valley_level
                    + layer_index * layer_height
                    + relative_z
                )

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
            stacked_surface_id = f"{prefix}{surface_id}"
            assembly.surfaces[stacked_surface_id] = {
                "vertices": [
                    point_ids[point_id]
                    for point_id in model._surface_vertices(surface_id)
                ],
            }
            if surface_id in model.surface_holes:
                assembly.surface_holes[stacked_surface_id] = [
                    [point_ids[point_id] for point_id in loop]
                    for loop in model.surface_holes[surface_id]
                ]

    interfaces = [
        _interface_diagnostics(
            assembly,
            models,
            lower_index,
            tolerance,
        )
        for lower_index in range(len(models) - 1)
    ]
    if require_upper_support:
        unsupported = [
            interface
            for interface in interfaces
            if interface["unmatched_upper_surface_ids"]
        ]
        if unsupported:
            interface = unsupported[0]
            raise ValueError(
                "Upper layer "
                f"{interface['upper_layer']} has contact triangles with no "
                f"matching adhesive surface on layer {interface['lower_layer']}: "
                f"{interface['unmatched_upper_surface_ids'][:5]}"
            )

    return {
        "model": assembly,
        "num_layers": len(models),
        "layer_height": layer_height,
        "layer_heights": layer_heights,
        "valley_levels": valley_levels,
        "interfaces": interfaces,
        "max_interface_error": max(
            (
                interface["max_interface_error"]
                for interface in interfaces
            ),
            default=0.0,
        ),
    }


def stack_layers(
    model: Cadder,
    num_layers: int = 4,
    tolerance: float = 1e-6,
) -> LayerStackResult:
    """Build a static assembly by repeating one solved hexagon layer.

    This compatibility API delegates to :func:`stack_mixed_layers`.
    """
    if isinstance(num_layers, bool) or int(num_layers) != num_layers:
        raise ValueError("num_layers must be an integer.")
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    mixed_result = stack_mixed_layers(
        [model] * num_layers,
        tolerance=tolerance,
    )
    return {
        "model": mixed_result["model"],
        "num_layers": mixed_result["num_layers"],
        "layer_height": mixed_result["layer_height"],
        "valley_level": mixed_result["valley_levels"][0],
        "max_interface_error": mixed_result["max_interface_error"],
    }


# Explicit pattern-specific alias for callers that prefer a descriptive name.
stack_simple_hexagon_layers = stack_layers
