"""Nest flat origami patterns on rectangular stock for DXF export."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
import math
from pathlib import Path
from typing import TYPE_CHECKING

from ezdxf import units
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from .._validation import finite, integer
from ._shared_cuts import merge_shared_cuts
from ._sheet_packing import pack_sheet
from .dxf_export import CreaseStyle, _dxf_unit, _finite_xy, save_dxf

if TYPE_CHECKING:
    from ..core.two_d_drawer import TwoDDrawer


_DEFAULT_ROTATIONS = (0.0, 90.0, 180.0, 270.0)
_DEFAULT_SHEET_CREASE_STYLE = ("real dashed", 2.0, 2.0)
_OUTLINE_TOLERANCE_MM = 1e-8
_Point = tuple[float, float]
_Segment = tuple[_Point, _Point]


@dataclass(frozen=True)
class SheetPattern:
    """A named flat drawing or its 2D metadata, with an optional copy limit.

    A non-negative integer requests that many copies. Zero skips the pattern.
    ``None`` fills available space after the numbered requests, taking turns
    among fill patterns.
    """

    name: str
    pattern: TwoDDrawer | dict
    quantity: int | None = None


@dataclass(frozen=True)
class SheetPlacement:
    """Transform for one copy from source coordinates converted to millimetres.

    If ``mirrored`` is true, negate the source x coordinate first. Rotate
    counterclockwise about the source origin, then add the translation.
    Bounds are ``(xmin, ymin, xmax, ymax)`` in sheet coordinates.
    """

    name: str
    rotation_deg: float
    translation_mm: _Point
    bounds_mm: tuple[float, float, float, float]
    mirrored: bool = False


@dataclass(frozen=True)
class _PatternGeometry:
    outline: Polygon | MultiPolygon
    segments: tuple[tuple[str, _Segment], ...]
    quantity: int | None


@dataclass(frozen=True)
class SheetLayout:
    """Placements on a sheet with origin at its lower-left corner."""

    width_mm: float
    height_mm: float
    margin_mm: float
    spacing_mm: float
    placements: tuple[SheetPlacement, ...]
    _patterns: dict[str, _PatternGeometry] = field(repr=False, compare=False)

    @property
    def counts(self) -> dict[str, int]:
        """Copies placed per pattern, including patterns with zero copies."""
        placed = Counter(placement.name for placement in self.placements)
        return {name: placed[name] for name in self._patterns}

    @property
    def unplaced(self) -> dict[str, int]:
        """Outstanding copies for patterns with a finite requested quantity."""
        counts = self.counts
        return {
            name: pattern.quantity - counts[name]
            for name, pattern in self._patterns.items()
            if pattern.quantity is not None and counts[name] < pattern.quantity
        }

    @property
    def utilization(self) -> float:
        """Fraction of the full sheet occupied by footprints, including voids."""
        area = sum(
            self._patterns[name].outline.area * count for name, count in self.counts.items()
        )
        return area / (self.width_mm * self.height_mm)

    def save_dxf(
        self,
        filename: str | Path,
        *,
        include_creases: bool = True,
        crease_style: CreaseStyle = _DEFAULT_SHEET_CREASE_STYLE,
        include_rigid: bool = False,
        allow_partial: bool = False,
    ) -> Path:
        """Write a millimetre DXF with cut, crease, and optional rigid layers.

        Creases default to separate 2 mm DXF line segments with 2 mm gaps
        between them. Use ``crease_style=("real dashed", dash_mm, gap_mm)``
        to change the lengths. The stock rectangle is implicit. Hole cuts
        are preserved, and a common cut between touching patterns is emitted
        once. Use ``allow_partial`` to export a layout with outstanding
        numbered requests.
        """
        unplaced = self.unplaced
        if unplaced and not allow_partial:
            missing = ", ".join(f"{name} ({count})" for name, count in unplaced.items())
            raise ValueError(f"The layout is missing requested copies: {missing}.")
        if not self.placements:
            raise ValueError("No patterns fit on the sheet.")

        included_kinds = {"side"}
        if include_creases:
            included_kinds.update(("valley", "mountain"))
        if include_rigid:
            included_kinds.add("rigid")
        cuts: list[_Segment] = []
        marks: list[tuple[str, _Segment]] = []
        for placement in self.placements:
            radians = math.radians(placement.rotation_deg)
            cosine, sine = math.cos(radians), math.sin(radians)
            if abs(cosine) < 1e-15:
                cosine = 0.0
            if abs(sine) < 1e-15:
                sine = 0.0
            cos_x, sin_x = (-cosine, -sine) if placement.mirrored else (cosine, sine)
            tx, ty = placement.translation_mm
            for kind, ((x0, y0), (x1, y1)) in self._patterns[placement.name].segments:
                if kind not in included_kinds:
                    continue
                segment = (
                    (cos_x * x0 - sine * y0 + tx, sin_x * x0 + cosine * y0 + ty),
                    (cos_x * x1 - sine * y1 + tx, sin_x * x1 + cosine * y1 + ty),
                )
                if kind == "side":
                    cuts.append(segment)
                else:
                    marks.append((kind, segment))

        if self.spacing_mm == 0.0:
            cuts = merge_shared_cuts(cuts)
        points, lines = {}, {}
        entities = chain((("side", segment) for segment in cuts), marks)
        for index, (kind, (start, end)) in enumerate(entities):
            start_id, end_id = 2 * index, 2 * index + 1
            points[start_id], points[end_id] = start, end
            lines[index] = {"start": start_id, "end": end_id, "kind": kind}

        return save_dxf(
            {"metadata": {"unit": "mm"}, "points": points, "lines": lines},
            filename,
            include_creases=include_creases,
            crease_style=crease_style,
            include_rigid=include_rigid,
        )


def layout_sheet(
    patterns: Sequence[SheetPattern],
    width_mm: float,
    height_mm: float,
    *,
    margin_mm: float = 0.0,
    spacing_mm: float = 0.0,
    rotations_deg: Sequence[float] = _DEFAULT_ROTATIONS,
    allow_mirror: bool = True,
) -> SheetLayout:
    """Find a compact, non-overlapping layout using the actual cut outlines.

    Sheet size, edge margin, and inter-pattern spacing are in millimetres.
    Inter-pattern spacing defaults to zero. The search can mirror copies
    before rotation. Set ``allow_mirror=False`` to preserve handedness.
    Each drawing can declare its own supported DXF length unit and is converted
    to millimetres before layout. Each needs a closed outer ``side`` contour.
    Cavities and punch holes are reserved within the footprint. The search is
    heuristic and reports any unplaced numbered copies on the result.
    """
    width_mm = finite("width_mm", width_mm, positive=True)
    height_mm = finite("height_mm", height_mm, positive=True)
    margin_mm = finite("margin_mm", margin_mm, nonnegative=True)
    spacing_mm = finite("spacing_mm", spacing_mm, nonnegative=True)
    if not isinstance(allow_mirror, bool):
        raise ValueError("allow_mirror must be a boolean.")
    if 2.0 * margin_mm >= min(width_mm, height_mm):
        raise ValueError("margin_mm must leave a positive usable sheet width and height.")
    rotations = tuple(dict.fromkeys(
        finite("rotation angle", angle) % 360 for angle in rotations_deg
    ))
    if not rotations:
        raise ValueError("At least one rotation angle is required.")

    geometries = {}
    for request in patterns:
        if not isinstance(request, SheetPattern):
            raise TypeError("patterns must contain SheetPattern objects.")
        if not isinstance(request.name, str) or not request.name.strip():
            raise ValueError("Each sheet pattern must have a non-empty name.")
        if request.name in geometries:
            raise ValueError(f"Sheet pattern name {request.name!r} is duplicated.")
        geometries[request.name] = _prepare_pattern(request)
    if not geometries:
        raise ValueError("At least one sheet pattern is required.")

    names = tuple(geometries)
    packed = pack_sheet(
        [geometry.outline for geometry in geometries.values()],
        [geometry.quantity for geometry in geometries.values()],
        width_mm,
        height_mm,
        margin=margin_mm,
        spacing=spacing_mm,
        rotations=rotations,
        allow_mirror=allow_mirror,
    )
    placements = tuple(
        SheetPlacement(
            names[item.pattern_index], item.rotation, item.translation, item.geometry.bounds,
            item.mirrored,
        )
        for item in packed
    )
    return SheetLayout(width_mm, height_mm, margin_mm, spacing_mm, placements, geometries)


def save_sheet_dxf(
    patterns: Sequence[SheetPattern],
    filename: str | Path,
    width_mm: float,
    height_mm: float,
    *,
    margin_mm: float = 0.0,
    spacing_mm: float = 0.0,
    rotations_deg: Sequence[float] = _DEFAULT_ROTATIONS,
    allow_mirror: bool = True,
    include_creases: bool = True,
    crease_style: CreaseStyle = _DEFAULT_SHEET_CREASE_STYLE,
    include_rigid: bool = False,
    allow_partial: bool = False,
) -> SheetLayout:
    """Lay out patterns, write one laser-cut DXF, and return the layout.

    See :meth:`SheetLayout.save_dxf` for the crease style and export options.
    """
    layout = layout_sheet(
        patterns,
        width_mm,
        height_mm,
        margin_mm=margin_mm,
        spacing_mm=spacing_mm,
        rotations_deg=rotations_deg,
        allow_mirror=allow_mirror,
    )
    layout.save_dxf(
        filename,
        include_creases=include_creases,
        crease_style=crease_style,
        include_rigid=include_rigid,
        allow_partial=allow_partial,
    )
    return layout


def _prepare_pattern(request: SheetPattern) -> _PatternGeometry:
    quantity = request.quantity
    if quantity is not None:
        quantity = integer(f"quantity for {request.name!r}", quantity, minimum=0)
    metadata = request.pattern
    if not isinstance(metadata, dict):
        metadata = metadata.to_dict()
    source_unit = _dxf_unit(metadata.get("metadata", {}).get("unit", "mm"))
    scale = units.conversion_factor(source_unit, units.MM)
    points = {}
    for point_id, coords in metadata.get("points", {}).items():
        x, y = _finite_xy(coords)
        points[point_id] = _finite_xy((x * scale, y * scale))

    segments = []
    cuts = []
    for line_id, line in metadata.get("lines", {}).items():
        kind = line.get("kind", "side")
        if kind == "construction":
            continue
        if kind not in {"side", "rigid", "mountain", "valley"}:
            raise ValueError(f"Pattern {request.name!r}, line {line_id!r} has unknown kind {kind!r}.")
        try:
            segment = points[line["start"]], points[line["end"]]
        except KeyError as error:
            raise ValueError(
                f"Pattern {request.name!r}, line {line_id!r} references a missing endpoint."
            ) from error
        if segment[0] == segment[1]:
            continue
        segments.append((kind, segment))
        if kind == "side":
            cuts.append(LineString(segment))

    faces = polygonize(unary_union(cuts))
    # Fill holes for collision detection while retaining their original cuts.
    outline = unary_union([Polygon(face.exterior) for face in faces])
    if outline.is_empty:
        raise ValueError(f"Pattern {request.name!r} needs a closed outer side contour.")
    coverage = outline.buffer(_OUTLINE_TOLERANCE_MM)
    for _, segment in segments:
        if not coverage.covers(LineString(segment)):
            raise ValueError(f"Pattern {request.name!r} has lines outside its closed side contours.")

    return _PatternGeometry(outline, tuple(segments), quantity)
