"""File exporters used by :class:`origamicad.Cadder`.

STL is written as a triangle mesh. STEP writes planar BREP faces as connected
surface shells or as separate thickened panel solids, without a CAD kernel.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from .._validation import finite

HoleLoopGeometry = tuple[tuple[str, ...], np.ndarray]
PanelGeometry = tuple[
    str,
    tuple[str, ...],
    np.ndarray,
    list[HoleLoopGeometry],
]
_STEP_LENGTH_TOLERANCE = 1e-7


def model_to_dict(model) -> dict:
    """Return the current 3D configuration as JSON-compatible metadata."""
    data = {
        "metadata": {
            "format": "OrigamiCAD 3D",
            "dimension": 3,
            "unit": model.unit,
        },
        "points": {
            pid: [float(point.x), float(point.y), float(point.z)]
            for pid, point in model.points.items()
        },
        "lines": {},
        "surfaces": {},
    }

    for line_id in model.lines:
        start, end, kind = model._line_info(line_id)
        data["lines"][line_id] = {
            "start": start,
            "end": end,
            "kind": kind,
        }

    for surface_id in model.surfaces:
        data["surfaces"][surface_id] = {
            "vertices": model._surface_vertices(surface_id),
        }

    if model.hex_units:
        data["hex_units"] = _json_ready(model.hex_units)
        data["hex_creases"] = _json_ready(model.hex_creases)
    if getattr(model, "surface_holes", None):
        data["surface_holes"] = _json_ready(model.surface_holes)

    return data


def save_json(model, filename: str | Path) -> Path:
    """Write the current 3D metadata to JSON."""
    path = Path(filename)
    with path.open("w", encoding="utf-8") as file:
        json.dump(model_to_dict(model), file, indent=2)
        file.write("\n")
    return path


def save_stl(
    model,
    filename: str | Path,
    thickness: float = 0.0,
    solid_name: str = "OrigamiCAD",
) -> Path:
    """Write an ASCII STL surface mesh or a collection of solid panels."""
    path = Path(filename)
    name = _safe_name(solid_name)
    triangles = _mesh_triangles(model, thickness)

    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write(f"solid {name}\n")
        for triangle in triangles:
            normal = _triangle_normal(triangle)
            file.write(
                "  facet normal "
                f"{_number(normal[0])} {_number(normal[1])} {_number(normal[2])}\n"
            )
            file.write("    outer loop\n")
            for vertex in triangle:
                file.write(
                    "      vertex "
                    f"{_number(vertex[0])} "
                    f"{_number(vertex[1])} "
                    f"{_number(vertex[2])}\n"
                )
            file.write("    endloop\n  endfacet\n")
        file.write(f"endsolid {name}\n")

    return path


def save_step(
    model,
    filename: str | Path,
    thickness: float = 0.0,
    model_name: str = "OrigamiCAD",
    *,
    separate_layer_parts: bool = False,
) -> Path:
    """Write STEP surfaces or advanced BREP solids without a CAD kernel.

    By default, zero-thickness exports join panels with shared model edges
    into surface shells and merge adjacent coplanar panels into single faces.
    Set ``separate_layer_parts=True`` for a stacked model whose surface IDs
    use the ``layer_<index>::<surface_id>`` convention.  The resulting STEP
    file contains one product definition per layer, with individual panel
    faces, allowing Abaqus/CAE to import the layers as separate parts.
    Hole punches in ``model.surface_holes`` become inner face boundaries
    for zero-thickness surfaces.
    """
    thickness = finite("thickness", thickness, nonnegative=True)
    if separate_layer_parts and thickness != 0.0:
        raise ValueError(
            "separate_layer_parts requires thickness=0.0 so each layer can "
            "be exported as one connected shell. Assign the shell thickness "
            "in Abaqus/CAE after import."
        )
    path = Path(filename)
    step = _StepFile()

    # AP214 defines the manifold surface representation used by the sheet export.
    use_ap214 = thickness == 0.0 and not separate_layer_parts
    if use_ap214:
        application = "core data for automotive mechanical design processes"
        protocol = "'automotive_design',2000"
        schema = "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }"
    else:
        application = "configuration controlled 3d designs of mechanical parts and assemblies"
        protocol = "'config_control_design',1994"
        schema = "CONFIG_CONTROL_DESIGN"
    app = step.add(f"APPLICATION_CONTEXT('{application}')")
    step.add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard',{protocol},{app})"
    )
    product_context = step.add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
    definition_context = step.add(
        f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')"
    )

    length_unit = step.add(_step_length_unit(model.unit))
    angle_unit = step.add(
        "(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))"
    )
    solid_angle_unit = step.add(
        "(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())"
    )
    uncertainty = step.add(
        "UNCERTAINTY_MEASURE_WITH_UNIT("
        f"LENGTH_MEASURE({_step_number(_STEP_LENGTH_TOLERANCE)}),{length_unit},"
        "'distance_accuracy_value','confusion accuracy')"
    )
    context = step.add(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3) "
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({uncertainty})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({length_unit},{angle_unit},{solid_angle_unit})) "
        "REPRESENTATION_CONTEXT('','3D'))"
    )

    origin = step.cartesian_point((0.0, 0.0, 0.0))
    z_direction = step.add("DIRECTION('',(0.,0.,1.))")
    x_direction = step.add("DIRECTION('',(1.,0.,0.))")
    placement = step.add(
        f"AXIS2_PLACEMENT_3D('',{origin},{z_direction},{x_direction})"
    )

    if separate_layer_parts:
        part_groups = _layer_part_groups(model)
    else:
        part_groups = [(str(model_name), list(_panel_geometry(model)))]

    products = []
    for part_name, panels in part_groups:
        escaped_part_name = _step_string(part_name)
        product = step.add(
            f"PRODUCT('{escaped_part_name}','{escaped_part_name}','',"
            f"({product_context}))"
        )
        products.append(product)
        if use_ap214:
            formation = step.add(f"PRODUCT_DEFINITION_FORMATION('','',{product})")
        else:
            formation = step.add(
                "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE("
                f"'','',{product},.NOT_KNOWN.)"
            )
        product_definition = step.add(
            f"PRODUCT_DEFINITION('design','',{formation},{definition_context})"
        )
        product_shape = step.add(
            f"PRODUCT_DEFINITION_SHAPE('','',{product_definition})"
        )
        representation = _step_panel_representation(
            step,
            panels,
            thickness=thickness,
            representation_name=part_name,
            placement=placement,
            context=context,
            separate_layer_parts=separate_layer_parts,
        )
        step.add(
            f"SHAPE_DEFINITION_REPRESENTATION({product_shape},{representation})"
        )

    step.add(
        f"PRODUCT_RELATED_PRODUCT_CATEGORY('part','',({_refs(products)}))"
    )
    step.write(path, model_name, schema)
    return path


def save_cad(
    model,
    filename: str | Path,
    thickness: float = 0.0,
    *,
    separate_layer_parts: bool = False,
) -> Path:
    """Select JSON, STL, or STEP export from the filename extension.

    ``separate_layer_parts`` is available only for STEP output.
    """
    path = Path(filename)
    extension = path.suffix.lower()
    if separate_layer_parts and extension not in {".step", ".stp"}:
        raise ValueError(
            "separate_layer_parts is supported only for STEP output."
        )
    if extension == ".json":
        return save_json(model, path)
    if extension == ".stl":
        return save_stl(model, path, thickness=thickness)
    if extension in {".step", ".stp"}:
        return save_step(
            model,
            path,
            thickness=thickness,
            separate_layer_parts=separate_layer_parts,
        )
    raise ValueError("Output extension must be .json, .stl, .step, or .stp.")


def _panel_geometry(
    model,
) -> Iterable[PanelGeometry]:
    surface_holes = getattr(model, "surface_holes", {})
    for surface_id in model.surfaces:
        point_ids = tuple(model._surface_vertices(surface_id))
        if len(point_ids) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")
        polygon = np.array(
            [model.point_array(point_id) for point_id in point_ids],
            dtype=float,
        )
        hole_loops = []
        for loop in surface_holes.get(surface_id, []):
            hole_point_ids = tuple(loop)
            if len(hole_point_ids) < 3:
                raise ValueError(
                    f"Hole in surface '{surface_id}' has fewer than 3 vertices."
                )
            hole_loops.append(
                (
                    hole_point_ids,
                    np.array(
                        [
                            model.point_array(point_id)
                            for point_id in hole_point_ids
                        ],
                        dtype=float,
                    ),
                )
            )
        yield surface_id, point_ids, polygon, hole_loops


def _layer_part_groups(
    model,
) -> list[tuple[str, list[PanelGeometry]]]:
    """Group stacked panel polygons into zero-based, contiguous layer parts."""
    groups: dict[int, list[PanelGeometry]] = {}
    layer_pattern = re.compile(r"^layer_(\d+)::(.+)$")

    for surface_id, point_ids, polygon, hole_loops in _panel_geometry(model):
        match = layer_pattern.fullmatch(surface_id)
        if match is None:
            raise ValueError(
                "separate_layer_parts requires every surface ID to match "
                "'layer_<index>::<surface_id>'; "
                f"got {surface_id!r}."
            )
        layer_index = int(match.group(1))

        boundary_points = [("Surface", point_id) for point_id in point_ids] + [
            ("Hole in", point_id)
            for hole_point_ids, _ in hole_loops
            for point_id in hole_point_ids
        ]
        for owner, point_id in boundary_points:
            point_match = layer_pattern.fullmatch(point_id)
            if point_match is None or int(point_match.group(1)) != layer_index:
                raise ValueError(
                    f"{owner} {surface_id!r} references point {point_id!r} "
                    "outside its layer."
                )

        groups.setdefault(layer_index, []).append(
            (match.group(2), point_ids, polygon, hole_loops)
        )

    if not groups:
        raise ValueError(
            "separate_layer_parts requires at least one stacked surface."
        )
    layer_indices = sorted(groups)
    if layer_indices != list(range(len(layer_indices))):
        raise ValueError(
            "Layer indices must be contiguous and start at zero; "
            f"got {layer_indices}."
        )
    return [
        (f"Layer_{layer_index + 1}", groups[layer_index])
        for layer_index in layer_indices
    ]


def _step_panel_representation(
    step,
    panels: Iterable[PanelGeometry],
    *,
    thickness: float,
    representation_name: str,
    placement: str,
    context: str,
    separate_layer_parts: bool,
) -> str:
    """Add one STEP shape representation containing the supplied panels."""
    escaped_name = _step_string(representation_name)
    panels = list(panels)

    if thickness == 0.0:
        shells = []
        groups = _panel_shell_groups(panels, require_single_shell=separate_layer_parts)
        for group in groups:
            coordinates, face_indices, face_names, face_inner_indices = (
                _shell_geometry(group)
            )
            if not separate_layer_parts and len(group) > 1:
                face_indices, face_names, face_inner_indices = _merge_coplanar_faces(
                    coordinates, face_indices, face_names, face_inner_indices
                )
            faces = step.advanced_faces(
                coordinates,
                face_indices,
                face_names,
                face_inner_indices=face_inner_indices,
            )
            shell_type = "OPEN_SHELL"
            if not separate_layer_parts and _is_closed_shell(face_indices, face_inner_indices):
                shell_type = "CLOSED_SHELL"
            shells.append(step.add(f"{shell_type}('',({_refs(faces)}))"))

        if not shells:
            return step.add(
                f"SHAPE_REPRESENTATION('{escaped_name}',({placement}),{context})"
            )

        surface_model = step.add(
            f"SHELL_BASED_SURFACE_MODEL('{escaped_name}',({_refs(shells)}))"
        )
        return step.add(
            f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('{escaped_name}',"
            f"({_refs([placement, surface_model])}),{context})"
        )

    if any(hole_loops for _, _, _, hole_loops in panels):
        raise ValueError(
            "STEP export of thickened panels with hole punches is not yet "
            "supported. Use thickness=0.0 for punched shell surfaces."
        )

    solids = []
    half = 0.5 * thickness
    for surface_id, _, polygon, _ in panels:
        normal = _polygon_normal(polygon)
        top = polygon + half * normal
        bottom = polygon - half * normal
        coordinates = np.vstack((top, bottom))
        count = len(polygon)

        face_indices = [
            list(range(count)),
            list(reversed(range(count, 2 * count))),
        ]
        face_names = [f"{surface_id} top", f"{surface_id} bottom"]
        for index in range(count):
            next_index = (index + 1) % count
            face_indices.append(
                [
                    count + index,
                    count + next_index,
                    next_index,
                    index,
                ]
            )
            face_names.append(f"{surface_id} side {index}")

        faces = step.advanced_faces(
            coordinates,
            face_indices,
            face_names,
        )
        shell = step.add(f"CLOSED_SHELL('',({_refs(faces)}))")
        solids.append(
            step.add(
                f"MANIFOLD_SOLID_BREP('{_step_string(surface_id)}',{shell})"
            )
        )

    return step.add(
        f"ADVANCED_BREP_SHAPE_REPRESENTATION('{escaped_name}',"
        f"({_refs([placement, *solids])}),{context})"
    )


def _panel_shell_groups(
    panels: list[PanelGeometry], *, require_single_shell: bool,
) -> list[list[PanelGeometry]]:
    """Group panels with shared edges and orient them consistently."""
    if require_single_shell and not panels:
        raise ValueError("A connected STEP shell requires at least one panel.")

    edge_occurrences: dict[
        tuple[str, str],
        list[tuple[int, bool]],
    ] = {}
    for face_index, (surface_id, point_ids, polygon, _) in enumerate(panels):
        if len(point_ids) != len(polygon):
            raise ValueError(
                f"Surface {surface_id!r} has inconsistent point metadata."
            )
        if len(set(point_ids)) != len(point_ids):
            raise ValueError(
                f"Surface {surface_id!r} contains duplicate point IDs."
            )

        for position, start_id in enumerate(point_ids):
            end_id = point_ids[(position + 1) % len(point_ids)]
            edge = tuple(sorted((start_id, end_id)))
            edge_occurrences.setdefault(edge, []).append(
                (face_index, (start_id, end_id) == edge)
            )

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in panels]
    branch_faces = set()
    for edge, occurrences in edge_occurrences.items():
        if len(occurrences) > 2:
            if require_single_shell:
                raise ValueError(
                    "Cannot create a manifold layer shell because edge "
                    f"{edge!r} belongs to {len(occurrences)} panels."
                )
            branch_faces.update(index for index, _ in occurrences)
        if len(occurrences) == 2:
            (first_face, first_direction), (second_face, second_direction) = occurrences
            flip_relation = first_direction == second_direction
            adjacency[first_face].append((second_face, flip_relation))
            adjacency[second_face].append((first_face, flip_relation))

    groups = []
    flips: dict[int, bool] = {}
    for first_face in range(len(panels)):
        if first_face in flips:
            continue
        if require_single_shell and groups:
            disconnected = [
                panel[0] for index, panel in enumerate(panels) if index not in flips
            ]
            raise ValueError(
                "Each exported layer must be connected through shared panel "
                f"edges; disconnected panels include {disconnected[:5]}."
            )

        flips[first_face] = False
        pending = [first_face]
        component = []
        orientable = True
        while pending:
            face_index = pending.pop()
            component.append(face_index)
            for neighbor, flip_relation in adjacency[face_index]:
                neighbor_flip = flips[face_index] ^ flip_relation
                if neighbor not in flips:
                    flips[neighbor] = neighbor_flip
                    pending.append(neighbor)
                elif flips[neighbor] != neighbor_flip:
                    orientable = False

        if not orientable or not branch_faces.isdisjoint(component):
            if require_single_shell:
                raise ValueError("Layer panels do not form an orientable STEP shell.")
            # Preserve the panel surfaces when their shared edges cannot form a manifold.
            groups.extend([panels[index]] for index in sorted(component))
            continue

        group = []
        for index in sorted(component):
            surface_id, point_ids, polygon, holes = panels[index]
            if flips[index]:
                point_ids = point_ids[::-1]
                polygon = polygon[::-1]
            group.append((surface_id, point_ids, polygon, holes))
        groups.append(group)
    return groups


def _is_closed_shell(face_indices, face_inner_indices) -> bool:
    """Check whether every edge has two incident face boundaries."""
    edge_counts = Counter(
        (min(start, end), max(start, end))
        for outer, holes in zip(face_indices, face_inner_indices)
        for loop in (outer, *holes)
        for start, end in zip(loop, loop[1:] + loop[:1])
    )
    return all(count == 2 for count in edge_counts.values())


def _shell_geometry(
    panels: list[PanelGeometry],
) -> tuple[
    np.ndarray,
    list[list[int]],
    list[str],
    list[list[list[int]]],
]:
    """Index the face boundaries, sharing vertices by model point ID."""

    point_indices: dict[str, int] = {}
    coordinates: list[np.ndarray] = []

    def shared_indices(loop_point_ids, loop_coordinates) -> list[int]:
        """Index a loop's vertices, reusing points already in the shell."""
        indices = []
        for point_id, coordinate in zip(loop_point_ids, loop_coordinates):
            if point_id not in point_indices:
                point_indices[point_id] = len(coordinates)
                coordinates.append(coordinate)
            elif not np.array_equal(coordinates[point_indices[point_id]], coordinate):
                raise ValueError(
                    f"Point {point_id!r} has inconsistent coordinates."
                )
            indices.append(point_indices[point_id])
        return indices

    face_indices = []
    face_names = []
    face_inner_indices = []

    for surface_id, point_ids, polygon, hole_loops in panels:
        face_indices.append(shared_indices(point_ids, polygon))
        face_names.append(surface_id)

        inner_indices = []
        normal = _polygon_normal(polygon)
        for hole_point_ids, hole_polygon in hole_loops:
            if len(hole_point_ids) != len(hole_polygon):
                raise ValueError(
                    f"Hole in {surface_id!r} has inconsistent point metadata."
                )
            distances = np.abs((hole_polygon - polygon[0]) @ normal)
            if float(np.max(distances)) > 1e-6:
                raise ValueError(
                    f"Hole in {surface_id!r} is not coplanar with its panel."
                )

            if float(np.dot(_polygon_normal(hole_polygon), normal)) < 0.0:
                hole_point_ids = tuple(reversed(hole_point_ids))
                hole_polygon = hole_polygon[::-1]

            inner_indices.append(shared_indices(hole_point_ids, hole_polygon))
        face_inner_indices.append(inner_indices)

    return np.array(coordinates), face_indices, face_names, face_inner_indices


def _merge_coplanar_faces(
    coordinates: np.ndarray,
    face_indices: list[list[int]],
    face_names: list[str],
    face_inner_indices: list[list[list[int]]],
) -> tuple[list[list[int]], list[str], list[list[list[int]]]]:
    """Remove shared seams between planar faces without changing fold edges."""
    if len(face_indices) < 2:
        return face_indices, face_names, face_inner_indices

    face_edges = [list(zip(loop, loop[1:] + loop[:1])) for loop in face_indices]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, edges in enumerate(face_edges):
        for start, end in edges:
            edge_faces.setdefault((min(start, end), max(start, end)), []).append(
                index
            )

    neighbors: list[list[int]] = [[] for _ in face_indices]
    for owners in edge_faces.values():
        if len(owners) == 2:
            first, second = owners
            neighbors[first].append(second)
            neighbors[second].append(first)

    normals = [_polygon_normal(coordinates[loop]) for loop in face_indices]
    all_vertices = [
        coordinates[outer + [vertex for loop in holes for vertex in loop]]
        for outer, holes in zip(face_indices, face_inner_indices)
    ]
    visited = set()
    merged_faces = {}
    removed = set()
    for first, normal in enumerate(normals):
        if first in visited:
            continue
        visited.add(first)
        origin = all_vertices[first][0]
        distances = (all_vertices[first] - origin) @ normal
        if np.max(np.abs(distances)) > _STEP_LENGTH_TOLERANCE:
            continue

        region = [first]
        pending = [first]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor in visited:
                    continue
                # A folded-over face may be coplanar with the opposite normal.
                if np.linalg.norm(normals[neighbor] - normal) > 1e-8:
                    continue
                distances = (all_vertices[neighbor] - origin) @ normal
                if np.max(np.abs(distances)) > _STEP_LENGTH_TOLERANCE:
                    continue
                visited.add(neighbor)
                region.append(neighbor)
                pending.append(neighbor)

        if len(region) == 1:
            continue
        boundary = _coplanar_region_boundaries(
            coordinates,
            normal,
            region,
            face_edges,
            face_indices,
            face_inner_indices,
            edge_faces,
        )
        if boundary is None:
            continue

        # The STEP writer derives the merged face's plane from its outer loop.
        polygon = coordinates[boundary[0]]
        try:
            merged_normal = _polygon_normal(polygon)
        except ValueError:
            continue
        if any(
            np.max(np.abs((all_vertices[index] - polygon[0]) @ merged_normal))
            > _STEP_LENGTH_TOLERANCE
            for index in region
        ):
            continue
        merged_faces[first] = boundary
        removed.update(region[1:])

    outer_loops, names, inner_loops = [], [], []
    for index, (outer, name, holes) in enumerate(
        zip(face_indices, face_names, face_inner_indices)
    ):
        if index not in removed:
            outer, holes = merged_faces.get(index, (outer, holes))
            outer_loops.append(outer)
            names.append(name)
            inner_loops.append(holes)
    return outer_loops, names, inner_loops


def _coplanar_region_boundaries(
    coordinates: np.ndarray,
    normal: np.ndarray,
    region: list[int],
    face_edges: list[list[tuple[int, int]]],
    face_indices: list[list[int]],
    face_holes: list[list[list[int]]],
    edge_faces: dict[tuple[int, int], list[int]],
) -> tuple[list[int], list[list[int]]] | None:
    """Trace a planar region's perimeter and retain every boundary vertex."""
    from shapely.geometry import Polygon

    region_indices = set(region)
    next_vertex = {}
    incoming = set()
    for index in region:
        for start, end in face_edges[index]:
            owners = edge_faces[(min(start, end), max(start, end))]
            if len(owners) == 2 and all(index in region_indices for index in owners):
                continue
            if start in next_vertex or end in incoming:
                return None
            next_vertex[start] = end
            incoming.add(end)
    if next_vertex.keys() != incoming:
        return None

    axis = int(np.argmax(np.abs(normal)))
    projection_axes = ((axis + 1) % 3, (axis + 2) % 3)
    origin = coordinates[face_indices[region[0]][0]]

    def project(loop):
        return (coordinates[loop] - origin)[:, projection_axes]

    outer_loops = []
    holes = [loop for index in region for loop in face_holes[index]]
    while next_vertex:
        start = next(iter(next_vertex))
        loop = [start]
        current = next_vertex.pop(start)
        while current != start:
            loop.append(current)
            current = next_vertex.pop(current)
        if len(loop) < 3:
            return None

        x, y = project(loop).T
        area_twice = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        if area_twice * normal[axis] > 0.0:
            outer_loops.append(loop)
        else:
            # STEP FACE_BOUND reverses the orientation of inner loops.
            holes.append(loop[::-1])

    if len(outer_loops) != 1:
        return None
    outer = outer_loops[0]
    merged = Polygon(project(outer), [project(loop) for loop in holes])
    if not merged.is_valid or not merged.boundary.is_simple:
        return None
    original_area = 0.0
    for index in region:
        original = Polygon(
            project(face_indices[index]),
            [project(hole) for hole in face_holes[index]],
        )
        if not original.is_valid:
            return None
        original_area += original.area
    if not np.isclose(merged.area, original_area, rtol=1e-10, atol=0.0):
        return None
    return outer, holes


def _mesh_triangles(model, thickness: float) -> list[np.ndarray]:
    thickness = finite("thickness", thickness, nonnegative=True)
    triangles = []

    for _, _, polygon, _ in _panel_geometry(model):
        if thickness == 0.0:
            triangles.extend(_triangulate_fan(polygon))
            continue

        normal = _polygon_normal(polygon)
        half = 0.5 * thickness
        top = polygon + half * normal
        bottom = polygon - half * normal
        triangles.extend(_triangulate_fan(top))
        triangles.extend(
            triangle[[0, 2, 1]]
            for triangle in _triangulate_fan(bottom)
        )

        for index in range(len(polygon)):
            next_index = (index + 1) % len(polygon)
            triangles.append(
                np.array(
                    [bottom[index], bottom[next_index], top[next_index]]
                )
            )
            triangles.append(
                np.array([bottom[index], top[next_index], top[index]])
            )

    return triangles


def _triangulate_fan(polygon: np.ndarray) -> list[np.ndarray]:
    """Triangulate a convex panel while preserving its vertex orientation."""
    return [
        np.array([polygon[0], polygon[index], polygon[index + 1]])
        for index in range(1, len(polygon) - 1)
    ]


def _polygon_normal(polygon: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=float)
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        normal += np.cross(point, next_point)
    magnitude = float(np.linalg.norm(normal))
    if magnitude < 1e-12:
        raise ValueError("Cannot export a degenerate panel with zero area.")
    return normal / magnitude


def _triangle_normal(triangle: np.ndarray) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    magnitude = float(np.linalg.norm(normal))
    if magnitude < 1e-12:
        raise ValueError("Cannot export a degenerate triangle with zero area.")
    return normal / magnitude


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return cleaned or "OrigamiCAD"


def _number(value: float) -> str:
    value = float(value)
    if abs(value) < 5e-15:
        value = 0.0
    return f"{value:.15g}"


def _step_number(value: float) -> str:
    """Use the decimal point and uppercase exponent required by STEP REALs."""
    mantissa, separator, exponent = _number(value).upper().partition("E")
    if "." not in mantissa:
        mantissa += "."
    return mantissa + separator + exponent


def _step_string(value: str) -> str:
    ascii_value = str(value).encode("ascii", errors="replace").decode("ascii")
    return ascii_value.replace("'", "''")


def _refs(references: Iterable[str]) -> str:
    return ",".join(references)


def _step_length_unit(unit: str) -> str:
    prefixes = {
        "m": "$",
        "meter": "$",
        "metre": "$",
        "mm": ".MILLI.",
        "millimeter": ".MILLI.",
        "millimetre": ".MILLI.",
        "cm": ".CENTI.",
        "centimeter": ".CENTI.",
        "centimetre": ".CENTI.",
    }
    key = str(unit).strip().lower()
    if key not in prefixes:
        raise ValueError(
            "Native STEP export supports metre, centimetre, and millimetre "
            f"coordinates; got unit={unit!r}."
        )
    return f"(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({prefixes[key]},.METRE.))"


class _StepFile:
    def __init__(self):
        self.entities: list[str] = []

    def add(self, entity: str) -> str:
        self.entities.append(entity)
        return f"#{len(self.entities)}"

    def cartesian_point(
        self,
        coordinates,
        cache: dict | None = None,
    ) -> str:
        key = tuple(float(value) for value in coordinates)
        if cache is not None and key in cache:
            return cache[key]
        values = ",".join(_step_number(value) for value in key)
        reference = self.add(f"CARTESIAN_POINT('',({values}))")
        if cache is not None:
            cache[key] = reference
        return reference

    def direction(self, vector) -> str:
        vector = np.asarray(vector, dtype=float)
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-12:
            raise ValueError("Cannot create STEP geometry from a zero-length edge.")
        values = ",".join(_step_number(value) for value in vector / magnitude)
        return self.add(f"DIRECTION('',({values}))")

    def advanced_faces(
        self,
        coordinates: np.ndarray,
        face_indices: list[list[int]],
        face_names: list[str],
        face_inner_indices: list[list[list[int]]] | None = None,
    ) -> list[str]:
        """Create planar ADVANCED_FACE entities with shared edge topology."""
        coordinates = np.asarray(coordinates, dtype=float)
        if len(face_indices) != len(face_names):
            raise ValueError("Each STEP face must have one name.")
        if face_inner_indices is None:
            face_inner_indices = [[] for _ in face_indices]
        if len(face_inner_indices) != len(face_indices):
            raise ValueError("Each STEP face must have one inner-loop list.")

        used_indices = {
            index
            for outer, holes in zip(face_indices, face_inner_indices)
            for loop in (outer, *holes)
            for index in loop
        }
        point_refs = {
            index: self.cartesian_point(coordinates[index])
            for index in sorted(used_indices)
        }
        vertex_refs = {
            index: self.add(f"VERTEX_POINT('',{point_ref})")
            for index, point_ref in point_refs.items()
        }
        edge_cache = {}
        faces = []

        def edge_loop(indices: list[int]) -> str:
            oriented_edges = []
            for position, start_index in enumerate(indices):
                end_index = indices[(position + 1) % len(indices)]
                key = tuple(sorted((start_index, end_index)))

                if key not in edge_cache:
                    edge_vector = coordinates[end_index] - coordinates[start_index]
                    edge_length = float(np.linalg.norm(edge_vector))
                    direction = self.direction(edge_vector)
                    vector = self.add(
                        f"VECTOR('',{direction},{_step_number(edge_length)})"
                    )
                    line = self.add(
                        f"LINE('',{point_refs[start_index]},{vector})"
                    )
                    edge = self.add(
                        "EDGE_CURVE("
                        f"'',{vertex_refs[start_index]},{vertex_refs[end_index]},"
                        f"{line},.T.)"
                    )
                    edge_cache[key] = (start_index, end_index, edge)

                stored_start, stored_end, edge = edge_cache[key]
                same_direction = (
                    stored_start == start_index and stored_end == end_index
                )
                orientation = ".T." if same_direction else ".F."
                oriented_edges.append(
                    self.add(f"ORIENTED_EDGE('',*,*,{edge},{orientation})")
                )

            return self.add(f"EDGE_LOOP('',({_refs(oriented_edges)}))")

        for indices, inner_loops, name in zip(
            face_indices,
            face_inner_indices,
            face_names,
        ):
            polygon = coordinates[indices]
            normal = _polygon_normal(polygon)
            reference = polygon[1] - polygon[0]
            axis = self.add(
                "AXIS2_PLACEMENT_3D("
                f"'',{point_refs[indices[0]]},"
                f"{self.direction(normal)},{self.direction(reference)})"
            )
            plane = self.add(f"PLANE('',{axis})")

            outer_loop = edge_loop(indices)
            bounds = [
                self.add(f"FACE_OUTER_BOUND('',{outer_loop},.T.)")
            ]
            for inner_indices in inner_loops:
                if len(inner_indices) < 3:
                    raise ValueError(
                        "A STEP inner face boundary needs at least 3 vertices."
                    )
                inner_loop = edge_loop(inner_indices)
                bounds.append(
                    self.add(f"FACE_BOUND('',{inner_loop},.F.)")
                )
            faces.append(
                self.add(
                    f"ADVANCED_FACE('{_step_string(name)}',"
                    f"({_refs(bounds)}),{plane},.T.)"
                )
            )

        return faces

    def write(self, path: Path, model_name: str, schema: str) -> None:
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        timestamp = timestamp.replace("+00:00", "Z")
        filename = _step_string(path.name)
        name = _step_string(model_name)
        with path.open("w", encoding="ascii", newline="\n") as file:
            file.write("ISO-10303-21;\nHEADER;\n")
            file.write("FILE_DESCRIPTION(('Origami advanced BREP model'),'2;1');\n")
            file.write(
                f"FILE_NAME('{filename}','{timestamp}',(''),(''),"
                f"'OrigamiCAD','OrigamiCAD','{name}');\n"
            )
            file.write(f"FILE_SCHEMA(('{schema}'));\n")
            file.write("ENDSEC;\nDATA;\n")
            for index, entity in enumerate(self.entities, start=1):
                file.write(f"#{index}={entity};\n")
            file.write("ENDSEC;\nEND-ISO-10303-21;\n")
