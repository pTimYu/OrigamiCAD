"""File exporters used by :class:`origamicad.Cadder`.

The exporters intentionally depend only on NumPy and the Python standard
library.  STL is written as a triangle mesh.  STEP is written as faceted BREP
geometry, which preserves each thickened origami panel as a separate solid.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


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

    Set ``separate_layer_parts=True`` for a stacked model whose surface IDs
    use the ``layer_<index>::<surface_id>`` convention.  The resulting STEP
    file contains one product definition per layer, allowing Abaqus/CAE to
    import the layers as separate parts.
    """
    thickness = _validate_thickness(thickness)
    if separate_layer_parts and thickness != 0.0:
        raise ValueError(
            "separate_layer_parts requires thickness=0.0 so each layer can "
            "be exported as one connected shell. Assign the shell thickness "
            "in Abaqus/CAE after import."
        )
    path = Path(filename)
    step = _StepFile()

    app = step.add(
        "APPLICATION_CONTEXT(" 
        "'configuration controlled 3d designs of mechanical parts and assemblies')"
    )
    step.add(
        "APPLICATION_PROTOCOL_DEFINITION(" 
        f"'international standard','config_control_design',1994,{app})"
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
        f"LENGTH_MEASURE(1.E-7),{length_unit}," 
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
            connected_open_shell=separate_layer_parts,
        )
        step.add(
            f"SHAPE_DEFINITION_REPRESENTATION({product_shape},{representation})"
        )

    step.add(
        f"PRODUCT_RELATED_PRODUCT_CATEGORY('part','',({_refs(products)}))"
    )
    step.write(path, model_name)
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
) -> Iterable[tuple[str, tuple[str, ...], np.ndarray]]:
    for surface_id in model.surfaces:
        point_ids = tuple(model._surface_vertices(surface_id))
        if len(point_ids) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")
        yield surface_id, point_ids, np.array(
            [model.point_array(point_id) for point_id in point_ids],
            dtype=float,
        )


def _panel_polygons(model) -> Iterable[tuple[str, np.ndarray]]:
    for surface_id, _, polygon in _panel_geometry(model):
        yield surface_id, polygon


def _layer_part_groups(
    model,
) -> list[
    tuple[
        str,
        list[tuple[str, tuple[str, ...], np.ndarray]],
    ]
]:
    """Group stacked panel polygons into zero-based, contiguous layer parts."""
    groups: dict[
        int,
        list[tuple[str, tuple[str, ...], np.ndarray]],
    ] = {}
    layer_pattern = re.compile(r"^layer_(\d+)::(.+)$")

    for surface_id, point_ids, polygon in _panel_geometry(model):
        match = layer_pattern.fullmatch(surface_id)
        if match is None:
            raise ValueError(
                "separate_layer_parts requires every surface ID to match "
                "'layer_<index>::<surface_id>'; "
                f"got {surface_id!r}."
            )
        layer_index = int(match.group(1))
        local_surface_id = match.group(2)

        for point_id in point_ids:
            point_match = layer_pattern.fullmatch(point_id)
            if (
                point_match is None
                or int(point_match.group(1)) != layer_index
            ):
                raise ValueError(
                    f"Surface {surface_id!r} references point {point_id!r} "
                    "outside its layer."
                )

        groups.setdefault(layer_index, []).append(
            (local_surface_id, point_ids, polygon)
        )

    if not groups:
        raise ValueError(
            "separate_layer_parts requires at least one stacked surface."
        )

    layer_indices = sorted(groups)
    expected_indices = list(range(len(layer_indices)))
    if layer_indices != expected_indices:
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
    panels: Iterable[tuple[str, tuple[str, ...], np.ndarray]],
    *,
    thickness: float,
    representation_name: str,
    placement: str,
    context: str,
    connected_open_shell: bool,
) -> str:
    """Add one STEP shape representation containing the supplied panels."""
    escaped_name = _step_string(representation_name)
    panels = list(panels)

    if thickness == 0.0:
        if connected_open_shell:
            coordinates, face_indices, face_names = (
                _connected_shell_geometry(panels)
            )
            faces = step.advanced_faces(
                coordinates,
                face_indices,
                face_names,
            )
            shells = [step.add(f"OPEN_SHELL('',({_refs(faces)}))")]
        else:
            shells = []
            for surface_id, _, polygon in panels:
                faces = step.advanced_faces(
                    polygon,
                    [list(range(len(polygon)))],
                    [surface_id],
                )
                shells.append(step.add(f"OPEN_SHELL('',({_refs(faces)}))"))

        surface_model = step.add(
            f"SHELL_BASED_SURFACE_MODEL('{escaped_name}',({_refs(shells)}))"
        )
        return step.add(
            f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('{escaped_name}',"
            f"({_refs([placement, surface_model])}),{context})"
        )

    solids = []
    half = 0.5 * thickness
    for surface_id, _, polygon in panels:
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


def _connected_shell_geometry(
    panels: list[tuple[str, tuple[str, ...], np.ndarray]],
) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """Build consistently oriented faces with shared vertices and edges."""
    if not panels:
        raise ValueError("A connected STEP shell requires at least one panel.")

    edge_occurrences: dict[
        tuple[str, str],
        list[tuple[int, bool]],
    ] = {}
    for face_index, (surface_id, point_ids, polygon) in enumerate(panels):
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

    adjacency: dict[int, list[tuple[int, bool]]] = {
        face_index: []
        for face_index in range(len(panels))
    }
    for edge, occurrences in edge_occurrences.items():
        if len(occurrences) > 2:
            raise ValueError(
                "Cannot create a manifold layer shell because edge "
                f"{edge!r} belongs to {len(occurrences)} panels."
            )
        if len(occurrences) == 2:
            (first_face, first_direction), (
                second_face,
                second_direction,
            ) = occurrences
            flip_relation = first_direction == second_direction
            adjacency[first_face].append((second_face, flip_relation))
            adjacency[second_face].append((first_face, flip_relation))

    flips: dict[int, bool] = {0: False}
    pending = [0]
    while pending:
        face_index = pending.pop()
        for neighbor, flip_relation in adjacency[face_index]:
            neighbor_flip = flips[face_index] ^ flip_relation
            if neighbor in flips:
                if flips[neighbor] != neighbor_flip:
                    raise ValueError(
                        "Layer panels do not form an orientable STEP shell."
                    )
                continue
            flips[neighbor] = neighbor_flip
            pending.append(neighbor)

    if len(flips) != len(panels):
        disconnected = [
            panels[index][0]
            for index in range(len(panels))
            if index not in flips
        ]
        raise ValueError(
            "Each exported layer must be connected through shared panel "
            "edges; disconnected panels include "
            f"{disconnected[:5]}."
        )

    point_indices: dict[str, int] = {}
    point_coordinates: dict[str, np.ndarray] = {}
    coordinates: list[np.ndarray] = []
    face_indices = []
    face_names = []

    for face_index, (surface_id, point_ids, polygon) in enumerate(panels):
        if flips[face_index]:
            point_ids = tuple(reversed(point_ids))
            polygon = polygon[::-1]

        indices = []
        for point_id, coordinate in zip(point_ids, polygon):
            if point_id not in point_indices:
                point_indices[point_id] = len(coordinates)
                point_coordinates[point_id] = coordinate
                coordinates.append(coordinate)
            elif not np.array_equal(point_coordinates[point_id], coordinate):
                raise ValueError(
                    f"Point {point_id!r} has inconsistent coordinates."
                )
            indices.append(point_indices[point_id])

        face_indices.append(indices)
        face_names.append(surface_id)

    return np.array(coordinates), face_indices, face_names


def _mesh_triangles(model, thickness: float) -> list[np.ndarray]:
    thickness = _validate_thickness(thickness)
    triangles = []

    for _, polygon in _panel_polygons(model):
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


def _validate_thickness(thickness: float) -> float:
    thickness = float(thickness)
    if not np.isfinite(thickness) or thickness < 0.0:
        raise ValueError("thickness must be a finite value greater than or equal to 0.")
    return thickness


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
        values = ",".join(_number(value) for value in key)
        reference = self.add(f"CARTESIAN_POINT('',({values}))")
        if cache is not None:
            cache[key] = reference
        return reference

    def direction(self, vector) -> str:
        vector = np.asarray(vector, dtype=float)
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-12:
            raise ValueError("Cannot create STEP geometry from a zero-length edge.")
        values = ",".join(_number(value) for value in vector / magnitude)
        return self.add(f"DIRECTION('',({values}))")

    def advanced_faces(
        self,
        coordinates: np.ndarray,
        face_indices: list[list[int]],
        face_names: list[str],
    ) -> list[str]:
        """Create planar ADVANCED_FACE entities with shared edge topology."""
        coordinates = np.asarray(coordinates, dtype=float)
        if len(face_indices) != len(face_names):
            raise ValueError("Each STEP face must have one name.")

        point_refs = [self.cartesian_point(point) for point in coordinates]
        vertex_refs = [
            self.add(f"VERTEX_POINT('',{point_ref})")
            for point_ref in point_refs
        ]
        edge_cache = {}
        faces = []

        for indices, name in zip(face_indices, face_names):
            polygon = coordinates[indices]
            normal = _polygon_normal(polygon)
            reference = polygon[1] - polygon[0]
            axis = self.add(
                "AXIS2_PLACEMENT_3D(" 
                f"'',{point_refs[indices[0]]},"
                f"{self.direction(normal)},{self.direction(reference)})"
            )
            plane = self.add(f"PLANE('',{axis})")

            oriented_edges = []
            for position, start_index in enumerate(indices):
                end_index = indices[(position + 1) % len(indices)]
                key = tuple(sorted((start_index, end_index)))

                if key not in edge_cache:
                    edge_vector = coordinates[end_index] - coordinates[start_index]
                    edge_length = float(np.linalg.norm(edge_vector))
                    direction = self.direction(edge_vector)
                    vector = self.add(
                        f"VECTOR('',{direction},{_number(edge_length)})"
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

            loop = self.add(f"EDGE_LOOP('',({_refs(oriented_edges)}))")
            bound = self.add(f"FACE_OUTER_BOUND('',{loop},.T.)")
            faces.append(
                self.add(
                    f"ADVANCED_FACE('{_step_string(name)}',({bound}),{plane},.T.)"
                )
            )

        return faces

    def write(self, path: Path, model_name: str) -> None:
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
            file.write("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n")
            file.write("ENDSEC;\nDATA;\n")
            for index, entity in enumerate(self.entities, start=1):
                file.write(f"#{index}={entity};\n")
            file.write("ENDSEC;\nEND-ISO-10303-21;\n")
