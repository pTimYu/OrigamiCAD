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
) -> str:
    """Convert 2D metadata to a complete AutoCAD 2000 DXF document.

    ``crease_style=["real dashed", a, b]`` emits actual continuous LINE
    entities of length ``a``, separated by empty spans of length ``b``.
    Both lengths use the unit declared by the input metadata.

    The only supported profile is ``"solidworks"``. It is retained as an
    argument for source compatibility.
    """
    crease_style_spec = _parse_crease_style(crease_style)
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
