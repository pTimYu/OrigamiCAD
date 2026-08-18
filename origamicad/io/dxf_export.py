from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ezdxf
from ezdxf import bbox, units


CreaseStyle = Literal["solid", "dashed"] | list[object] | tuple[object, ...]
DxfProfile = Literal["solidworks"]
_MAX_REAL_DASH_SEGMENTS = 1_000_000
_DASH_LENGTH_MM = 3.0
_CONNECTING_DOT_LENGTH_MM = 0.6
_CONNECTING_DOT_COVERAGE = 0.01


@dataclass(frozen=True)
class _CreaseStyleSpec:
    name: Literal["solid", "dashed", "real dashed"]
    dash_length: float | None = None
    gap_length: float | None = None


def save_dxf(
    metadata: dict,
    filename: str | Path,
    include_creases: bool = True,
    crease_style: CreaseStyle = "dashed",
    include_construction: bool = False,
    include_rigid: bool = True,
    include_side: bool = True,
    profile: DxfProfile = "solidworks",
    connecting_dots: bool = False,
) -> Path:
    """Save 2D metadata as an AutoCAD/SolidWorks-compatible ASCII DXF file."""
    path = Path(filename)
    path.write_text(
        dxf_string_from_metadata(
            metadata,
            include_creases=include_creases,
            crease_style=crease_style,
            include_construction=include_construction,
            include_rigid=include_rigid,
            include_side=include_side,
            profile=profile,
            connecting_dots=connecting_dots,
        ),
        encoding="ascii",
        newline="",
    )
    return path


def dxf_string_from_metadata(
    metadata: dict,
    include_creases: bool = True,
    crease_style: CreaseStyle = "dashed",
    include_construction: bool = False,
    include_rigid: bool = True,
    include_side: bool = True,
    profile: DxfProfile = "solidworks",
    connecting_dots: bool = False,
) -> str:
    """Convert 2D metadata to a complete AutoCAD 2000 DXF document.

    ``crease_style=["real dashed", a, b]`` emits actual continuous LINE
    entities of length ``a``, separated by empty spans of length ``b``.
    Both lengths use the unit declared by the input metadata.

    Set ``connecting_dots=True`` to replace the closed side boundaries with
    evenly distributed 0.6 mm blank gaps while keeping the remaining boundary
    as solid cut lines. The total blank length is calculated to cover
    approximately 1% of the outer figure's perimeter. Inner cavity and
    hole-punch boundaries remain continuous.

    The only supported profile is ``"solidworks"``. It is retained as an
    argument for source compatibility.
    """
    crease_style_spec = _parse_crease_style(crease_style)
    if not isinstance(connecting_dots, bool):
        raise ValueError("connecting_dots must be a boolean.")
    if profile != "solidworks":
        raise ValueError(
            "The 'standard' DXF profile has been removed because it emitted "
            "invalid AC1015 tables. The only supported profile is 'solidworks'."
        )

    source_unit = metadata.get("metadata", {}).get("unit", "mm")
    document_unit = _dxf_unit(source_unit)
    points = metadata.get("points", {})
    lines = metadata.get("lines", {})

    document = ezdxf.new("R2000", setup=False, units=document_unit)
    dash_length = _DASH_LENGTH_MM * units.conversion_factor(
        units.MM,
        document_unit,
    )
    connecting_dot_data = None
    if connecting_dots and include_side:
        connecting_dot_data = _create_connecting_dots(
            metadata,
            dot_length=_CONNECTING_DOT_LENGTH_MM
            * units.conversion_factor(units.MM, document_unit),
            coverage=_CONNECTING_DOT_COVERAGE,
        )
    document.linetypes.add(
        "DASHED",
        pattern=[
            2.0 * dash_length,
            dash_length,
            -dash_length,
        ],
        description="Dashed __ __ __",
    )
    for layer, color, linetype in _layer_defs(
        crease_style_spec,
        include_creases=include_creases,
        include_construction=include_construction,
        include_rigid=include_rigid,
        include_side=include_side,
    ):
        document.layers.add(layer, color=color, linetype=linetype)

    modelspace = document.modelspace()

    for line_id, line in lines.items():
        kind = line.get("kind", "side")
        if kind not in {"valley", "mountain", "construction", "rigid", "side"}:
            kind = "side"

        if kind in {"valley", "mountain"} and not include_creases:
            continue
        if kind == "construction" and not include_construction:
            continue
        if kind == "rigid" and not include_rigid:
            continue
        if kind == "side" and not include_side:
            continue

        if (
            kind == "side"
            and connecting_dot_data is not None
            and line_id in connecting_dot_data[1]
        ):
            continue

        start = line["start"]
        end = line["end"]
        if start not in points or end not in points:
            raise ValueError(
                f"Line '{line_id}' references missing point(s): {start}, {end}."
            )

        if (
            kind in {"valley", "mountain"}
            and crease_style_spec.name == "real dashed"
        ):
            assert crease_style_spec.dash_length is not None
            assert crease_style_spec.gap_length is not None
            for dash_start, dash_end in _real_dash_segments(
                points[start],
                points[end],
                dash_length=crease_style_spec.dash_length,
                gap_length=crease_style_spec.gap_length,
            ):
                _add_line_entity(
                    modelspace,
                    dash_start,
                    dash_end,
                    kind,
                    crease_style_spec,
                )
            continue

        _add_line_entity(
            modelspace,
            points[start],
            points[end],
            kind,
            crease_style_spec,
        )

    if connecting_dot_data is not None:
        for solid_start, solid_end in connecting_dot_data[0]:
            _add_line_entity(
                modelspace,
                solid_start,
                solid_end,
                "side",
                crease_style_spec,
            )

    extents = bbox.extents(modelspace, fast=True)
    if extents.has_data:
        document.header["$EXTMIN"] = extents.extmin
        document.header["$EXTMAX"] = extents.extmax
        document.header["$LIMMIN"] = extents.extmin.vec2
        document.header["$LIMMAX"] = extents.extmax.vec2

    stream = io.StringIO()
    document.write(stream, fmt="asc")
    # CRLF is the conventional ASCII DXF line ending and remains the most
    # conservative choice for Windows CAD applications.
    return stream.getvalue().replace("\r\n", "\n").replace("\n", "\r\n")


def _add_line_entity(
    modelspace,
    start,
    end,
    kind: str,
    crease_style: _CreaseStyleSpec,
) -> None:
    x0, y0 = _finite_xy(start)
    x1, y1 = _finite_xy(end)
    layer, color, linetype = _line_properties(kind, crease_style)
    modelspace.add_line(
        (x0, y0),
        (x1, y1),
        dxfattribs={
            "layer": layer,
            "color": color,
            "linetype": linetype,
        },
    )


def _create_connecting_dots(
    metadata: dict,
    *,
    dot_length: float,
    coverage: float,
) -> tuple[
    list[tuple[tuple[float, float], tuple[float, float]]],
    set[str],
]:
    """Create solid cut segments with blank connecting dots.

    Side lines are stored as independent segments, so this first reconstructs
    planar loops and selects the loop with the largest enclosed area as the
    figure boundary. Blank dots are then placed at equal arc-length intervals
    on that loop, and the complementary solid segments are returned. Returning
    the source line IDs lets the exporter replace only the boundary lines it
    has successfully reconstructed; cavity, hole-punch, and unrelated side
    lines are preserved.
    """
    if not math.isfinite(dot_length) or dot_length <= 0.0:
        raise ValueError("dot_length must be a positive finite number.")
    if not math.isfinite(coverage) or coverage <= 0.0:
        raise ValueError("coverage must be a positive finite number.")

    points = metadata.get("points", {})
    lines = metadata.get("lines", {})
    excluded_line_ids = _hole_punch_line_ids(metadata, points)
    paths = _closed_side_paths(points, lines, excluded_line_ids)
    if not paths:
        return [], set()

    outer_path, outer_line_ids = max(
        paths,
        key=lambda path_data: abs(
            _signed_area(
                [_finite_xy(points[start]) for start, _ in path_data[0]]
            )
        ),
    )
    outer_length = _path_length(outer_path, points)
    if outer_length < dot_length:
        return [], set()

    dot_count = max(1, int(round(outer_length * coverage / dot_length)))
    dot_count = min(dot_count, _MAX_REAL_DASH_SEGMENTS)

    dot_centers = _dash_centers(
        outer_path,
        points,
        outer_length,
        dot_count,
        dot_length,
    )
    solid_intervals = []
    solid_start = 0.0
    for center in dot_centers:
        dot_start = center - 0.5 * dot_length
        dot_end = center + 0.5 * dot_length
        if dot_start > solid_start:
            solid_intervals.append((solid_start, dot_start))
        solid_start = dot_end
    if solid_start < outer_length:
        solid_intervals.append((solid_start, outer_length))

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    replaced_line_ids: set[str] = set()
    replaced_line_ids.update(outer_line_ids)
    for solid_start, solid_end in solid_intervals:
        segments.extend(
            _path_subsegments(
                outer_path,
                points,
                solid_start,
                solid_end,
            )
        )

    return segments, replaced_line_ids


def _hole_punch_line_ids(metadata: dict, points: dict) -> set[str]:
    """Return side-line IDs belonging to explicit hole-punch polygons."""
    hole_point_ids: set[str] = set()
    for hole in metadata.get("hole_punches", []):
        hole_id = hole.get("id")
        if hole_id is None:
            continue
        prefix = f"{hole_id}_p"
        hole_point_ids.update(
            point_id
            for point_id in points
            if str(point_id).startswith(prefix)
        )

    excluded: set[str] = set()
    for line_id, line in metadata.get("lines", {}).items():
        if (
            line.get("kind", "side") == "side"
            and line.get("start") in hole_point_ids
            and line.get("end") in hole_point_ids
        ):
            excluded.add(line_id)
    return excluded


def _closed_side_paths(
    points: dict,
    lines: dict,
    excluded_line_ids: set[str],
) -> list[tuple[list[tuple[str, str]], set[str]]]:
    """Trace the bounded faces formed by side-line segments.

    The half-edge walk handles both ordinary polygons and side graphs with
    vertices shared by several boundary segments. Positive signed-area walks
    are the bounded faces; the reverse walk of each simple loop is the outer
    face and is discarded.
    """
    adjacency: dict[str, list[tuple[str, str]]] = {}
    coordinates: dict[str, tuple[float, float]] = {}
    side_edges: list[tuple[str, str, str]] = []

    for line_id, line in lines.items():
        if line_id in excluded_line_ids:
            continue
        if line.get("kind", "side") != "side":
            continue

        start = line.get("start")
        end = line.get("end")
        if start not in points or end not in points:
            raise ValueError(
                f"Line '{line_id}' references missing point(s): {start}, {end}."
            )
        if start == end:
            continue

        coordinates[start] = _finite_xy(points[start])
        coordinates[end] = _finite_xy(points[end])
        side_edges.append((start, end, line_id))
        adjacency.setdefault(start, []).append((end, line_id))
        adjacency.setdefault(end, []).append((start, line_id))

    if not side_edges:
        return []

    for point_id, neighbors in adjacency.items():
        x0, y0 = coordinates[point_id]
        neighbors.sort(
            key=lambda item: math.atan2(
                coordinates[item[0]][1] - y0,
                coordinates[item[0]][0] - x0,
            )
        )

    directed_edges = [
        directed_edge
        for start, end, line_id in side_edges
        for directed_edge in (
            (start, end, line_id),
            (end, start, line_id),
        )
    ]
    visited: set[tuple[str, str, str]] = set()
    paths: list[tuple[list[tuple[str, str]], set[str]]] = []

    for first_edge in directed_edges:
        if first_edge in visited:
            continue

        current = first_edge
        walk: list[tuple[str, str, str]] = []
        closed = False
        while current not in visited:
            visited.add(current)
            walk.append(current)
            start, end, line_id = current

            neighbors = adjacency[end]
            reverse_index = next(
                (
                    index
                    for index, (neighbor, edge_id) in enumerate(neighbors)
                    if neighbor == start and edge_id == line_id
                ),
                None,
            )
            if reverse_index is None:
                break

            # The clockwise neighbor from the reverse half-edge keeps the
            # face on the left side of the walk.
            next_start, next_line_id = neighbors[
                (reverse_index - 1) % len(neighbors)
            ]
            current = (end, next_start, next_line_id)
            if current == first_edge:
                closed = True
                break

        if not closed or len(walk) < 3:
            continue

        polygon = [coordinates[start] for start, _, _ in walk]
        area = _signed_area(polygon)
        if area <= 0.0:
            continue

        paths.append(
            (
                [(start, end) for start, end, _ in walk],
                {line_id for _, _, line_id in walk},
            )
        )

    return paths


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(
            polygon,
            polygon[1:] + polygon[:1],
        )
    )


def _path_length(path: list[tuple[str, str]], points: dict) -> float:
    return sum(
        math.hypot(
            _finite_xy(points[end])[0] - _finite_xy(points[start])[0],
            _finite_xy(points[end])[1] - _finite_xy(points[start])[1],
        )
        for start, end in path
    )


def _dash_centers(
    path: list[tuple[str, str]],
    points: dict,
    path_length: float,
    count: int,
    dot_length: float,
) -> list[float]:
    """Choose evenly spaced dash centers away from polygon vertices."""
    spacing = path_length / count
    vertex_distances = []
    distance = 0.0
    for start_id, end_id in path[:-1]:
        x0, y0 = _finite_xy(points[start_id])
        x1, y1 = _finite_xy(points[end_id])
        distance += math.hypot(x1 - x0, y1 - y0)
        vertex_distances.append(distance)

    # A dash crossing a corner would be emitted as two shorter LINE entities.
    # Shift the regular pattern by a small deterministic amount until every
    # dash remains on one straight side whenever the geometry permits it.
    phase = 0.5 * spacing
    tolerance = max(path_length, dot_length) * 1e-12
    for _ in range(100):
        centers = [phase + index * spacing for index in range(count)]
        stays_inside_path = (
            centers[0] - 0.5 * dot_length >= -tolerance
            and centers[-1] + 0.5 * dot_length
            <= path_length + tolerance
        )
        if stays_inside_path and all(
            not any(
                center - 0.5 * dot_length - tolerance
                <= vertex_distance
                <= center + 0.5 * dot_length + tolerance
                for vertex_distance in vertex_distances
            )
            for center in centers
        ):
            return centers
        phase = (phase + 0.5 * dot_length) % spacing

    return [phase + index * spacing for index in range(count)]


def _path_subsegments(
    path: list[tuple[str, str]],
    points: dict,
    start_distance: float,
    end_distance: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the pieces of an arc-length interval as LINE segments."""
    if end_distance <= start_distance:
        return []

    segments = []
    distance = 0.0
    for start_id, end_id in path:
        x0, y0 = _finite_xy(points[start_id])
        x1, y1 = _finite_xy(points[end_id])
        length = math.hypot(x1 - x0, y1 - y0)
        segment_start = max(start_distance, distance)
        segment_end = min(end_distance, distance + length)
        if segment_end > segment_start and length > 0.0:
            t0 = (segment_start - distance) / length
            t1 = (segment_end - distance) / length
            segments.append(
                (
                    (x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0)),
                    (x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0)),
                )
            )
        distance += length
        if distance >= end_distance:
            break
    return segments


def _line_properties(
    kind: str,
    crease_style: _CreaseStyleSpec,
) -> tuple[str, int, str]:
    crease_linetype = (
        "DASHED" if crease_style.name == "dashed" else "CONTINUOUS"
    )
    if kind == "valley":
        return "CREASE_VALLEY", 5, crease_linetype
    if kind == "mountain":
        return "CREASE_MOUNTAIN", 1, crease_linetype
    if kind == "rigid":
        return "RIGID", 8, "CONTINUOUS"
    if kind == "construction":
        return "CONSTRUCTION", 9, "DASHED"
    return "CUT_SIDE", 7, "CONTINUOUS"


def _layer_defs(
    crease_style: _CreaseStyleSpec,
    *,
    include_creases: bool,
    include_construction: bool,
    include_rigid: bool,
    include_side: bool,
) -> list[tuple[str, int, str]]:
    layers = []
    if include_side:
        layers.append(("CUT_SIDE", 7, "CONTINUOUS"))
    if include_rigid:
        layers.append(("RIGID", 8, "CONTINUOUS"))
    if include_creases:
        crease_linetype = (
            "DASHED" if crease_style.name == "dashed" else "CONTINUOUS"
        )
        layers.extend(
            [
                ("CREASE_VALLEY", 5, crease_linetype),
                ("CREASE_MOUNTAIN", 1, crease_linetype),
            ]
        )
    if include_construction:
        layers.append(("CONSTRUCTION", 9, "DASHED"))
    return layers


def _parse_crease_style(crease_style: CreaseStyle) -> _CreaseStyleSpec:
    if isinstance(crease_style, str):
        if crease_style in {"solid", "dashed"}:
            return _CreaseStyleSpec(crease_style)
        raise ValueError(
            "crease_style must be 'solid', 'dashed', or "
            "['real dashed', dash_length, gap_length]."
        )

    if not isinstance(crease_style, (list, tuple)):
        raise ValueError(
            "crease_style must be 'solid', 'dashed', or "
            "['real dashed', dash_length, gap_length]."
        )
    if len(crease_style) != 3 or crease_style[0] != "real dashed":
        raise ValueError(
            "A real dashed crease_style must have the form "
            "['real dashed', dash_length, gap_length]."
        )

    dash_length = _positive_finite_length(crease_style[1], "dash_length")
    gap_length = _positive_finite_length(crease_style[2], "gap_length")
    return _CreaseStyleSpec("real dashed", dash_length, gap_length)


def _positive_finite_length(value, name: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{name} must be a positive finite number; got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a positive finite number; got {value!r}."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number; got {value!r}.")
    return result


def _real_dash_segments(
    start,
    end,
    *,
    dash_length: float,
    gap_length: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Split a crease into centered, continuous LINE segments.

    Full dashes and internal gaps retain their requested lengths. Any
    remainder is divided equally between the two endpoint gaps, so a dash
    never touches a side endpoint.
    """
    x0, y0 = _xy(start)
    x1, y1 = _xy(end)
    for value in (x0, y0, x1, y1):
        if not math.isfinite(value):
            raise ValueError(f"DXF coordinate must be finite, got {value!r}.")

    dx = x1 - x0
    dy = y1 - y0
    line_length = math.hypot(dx, dy)
    if not math.isfinite(line_length):
        raise ValueError("DXF line length must be finite.")

    period = dash_length + gap_length
    tolerance = (
        max(line_length, dash_length, gap_length) * 1e-12
        + max(
            math.ulp(line_length),
            math.ulp(dash_length),
            math.ulp(gap_length),
        )
        * 8.0
    )

    # A full dash must fit while leaving a non-zero gap at both endpoints.
    if line_length <= dash_length + 2.0 * tolerance:
        return []

    estimated_count = (line_length + gap_length) / period
    if not math.isfinite(estimated_count):
        raise ValueError(
            "The requested real dashed style would create more than "
            f"{_MAX_REAL_DASH_SEGMENTS:,} LINE entities for one crease."
        )

    dash_count = int(math.floor(estimated_count))
    if dash_count > _MAX_REAL_DASH_SEGMENTS:
        raise ValueError(
            "The requested real dashed style would create more than "
            f"{_MAX_REAL_DASH_SEGMENTS:,} LINE entities for one crease."
        )
    occupied_length = (
        dash_count * dash_length + max(0, dash_count - 1) * gap_length
    )

    # Exact division could otherwise put a solid dash directly on each side.
    # Removing one dash makes the endpoint gaps adaptive while keeping the
    # requested lengths for every emitted dash and internal gap.
    if line_length - occupied_length <= 2.0 * tolerance:
        dash_count -= 1
        occupied_length = (
            dash_count * dash_length + max(0, dash_count - 1) * gap_length
        )
    if dash_count <= 0:
        return []

    endpoint_gap = (line_length - occupied_length) / 2.0
    ux = dx / line_length
    uy = dy / line_length
    segments = []
    for index in range(dash_count):
        start_distance = endpoint_gap + index * period
        end_distance = start_distance + dash_length
        segments.append(
            (
                (x0 + ux * start_distance, y0 + uy * start_distance),
                (x0 + ux * end_distance, y0 + uy * end_distance),
            )
        )
    return segments


def _xy(coords) -> tuple[float, float]:
    if len(coords) < 2:
        raise ValueError(f"Point coordinate must contain at least x and y: {coords}")
    return float(coords[0]), float(coords[1])


def _finite_xy(coords) -> tuple[float, float]:
    x, y = _xy(coords)
    for value in (x, y):
        if not math.isfinite(value):
            raise ValueError(f"DXF coordinate must be finite, got {value!r}.")
    return x, y


def _dxf_unit(unit: str) -> int:
    key = str(unit).strip().lower()
    unit_codes = {
        "in": units.IN,
        "inch": units.IN,
        "inches": units.IN,
        "ft": units.FT,
        "foot": units.FT,
        "feet": units.FT,
        "mi": units.MI,
        "mile": units.MI,
        "miles": units.MI,
        "mm": units.MM,
        "millimeter": units.MM,
        "millimeters": units.MM,
        "cm": units.CM,
        "centimeter": units.CM,
        "centimeters": units.CM,
        "m": units.M,
        "meter": units.M,
        "meters": units.M,
        "km": units.KM,
        "kilometer": units.KM,
        "kilometers": units.KM,
        "yd": units.YD,
        "yard": units.YD,
        "yards": units.YD,
    }
    if key not in unit_codes:
        raise ValueError(
            "DXF export requires a known length unit; "
            f"got unit={unit!r}."
        )
    return unit_codes[key]
