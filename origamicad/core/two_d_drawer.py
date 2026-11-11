from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D as MplLine2D
from matplotlib.patches import Circle

from .._validation import finite, integer

LineKind = Literal["valley", "mountain", "side", "rigid", "construction"]

# Matplotlib style of every line kind, and the legend labels in legend order.
LINE_STYLES = {
    "valley": {"color": "blue", "linestyle": "--", "linewidth": 1.8},
    "mountain": {"color": "red", "linestyle": "-.", "linewidth": 1.8},
    "side": {"color": "black", "linestyle": "-", "linewidth": 2.2},
    "rigid": {"color": "black", "linestyle": "-", "linewidth": 1.0},
    "construction": {"color": "gray", "linestyle": ":", "linewidth": 1.0},
}
_LINE_LABELS = {
    "side": "Side",
    "valley": "Valley crease",
    "mountain": "Mountain crease",
    "rigid": "Rigid line",
    "construction": "Construction line",
}


def line_style(kind: str) -> dict:
    """Return the drawing style shared by the 2D and 3D previews."""
    try:
        return LINE_STYLES[kind]
    except KeyError:
        raise ValueError(f"Unknown line kind: {kind}") from None


def add_line_legend(ax, kinds) -> None:
    """Add a legend for the line kinds that are present."""
    handles = [
        MplLine2D([0], [0], label=label, **LINE_STYLES[kind])
        for kind, label in _LINE_LABELS.items()
        if kind in kinds
    ]
    if handles:
        ax.legend(handles=handles, loc="best")


@dataclass
class Point2D:
    id: str
    x: float
    y: float


@dataclass
class Line2D:
    id: str
    start: str
    end: str
    kind: LineKind = "side"


@dataclass
class Surface2D:
    id: str
    vertices: List[str]


class TwoDDrawer:
    """
    2D origami/kirigami pattern drawer.

    Core objects:
        point   -> 2D vertex
        line    -> valley crease, mountain crease, side, rigid line, or construction line
        surface -> rigid physical panel

    Notes:
        - There is no "void" surface type.
        - A void/cavity is represented by simply not adding a surface there.
        - Boundary lines around a void can still be drawn as "side" lines.
    """

    def __init__(self, unit: str = "mm", point_tol: float = 1e-9):
        self.unit = unit
        self.point_tol = point_tol

        self.points: Dict[str, Point2D] = {}
        self.lines: Dict[str, Line2D] = {}
        self.surfaces: Dict[str, Surface2D] = {}
        self.hole_punches: list[dict] = []
        self.surface_holes: dict[str, list[list[str]]] = {}
        self.hex_creases: dict[str, dict] = {}

        self._point_count = 0
        self._line_count = 0
        self._surface_count = 0
        self._geometry_indexes = None
        self._point_index = None

    @contextmanager
    def _index_geometry(self):
        """Index append-only layout construction, then discard the indexes.

        Public geometry stays mutable: later direct dictionary or vertex edits
        are observed by ordinary lookups and by the next layout operation.
        """
        if self._geometry_indexes is not None:
            yield
            return
        line_index, surface_index = {}, {}
        for lid, line in self.lines.items():
            line_index.setdefault(self._canonical_line_key(line.start, line.end), lid)
        for sid, surface in self.surfaces.items():
            surface_index.setdefault(self._canonical_surface_key(surface.vertices), sid)
        point_index = None
        if math.isfinite(self.point_tol) and self.point_tol > 0:
            buckets = {}
            for order, (pid, point) in enumerate(self.points.items()):
                bucket = self._point_bucket(point.x, point.y, self.point_tol)
                if bucket is not None:
                    buckets.setdefault(bucket, []).append((order, pid))
            point_index = self.point_tol, buckets
        self._geometry_indexes = line_index, surface_index
        self._point_index = point_index
        try:
            yield
        finally:
            self._geometry_indexes = None
            self._point_index = None

    @staticmethod
    def _point_bucket(x, y, width):
        """Return a spatial cell, or use ordinary lookup for extreme values."""
        scaled_x, scaled_y = x / width, y / width
        if not (math.isfinite(scaled_x) and math.isfinite(scaled_y)):
            return None
        return math.floor(scaled_x), math.floor(scaled_y)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "TwoDDrawer":
        """
        Build a 2D drawer from metadata produced by to_dict() or save_json().
        """
        info = metadata.get("metadata", {})
        drawer = cls(
            unit=info.get("unit", "mm"),
            point_tol=info.get("point_tol", 1e-9),
        )

        for point_id, coords in metadata.get("points", {}).items():
            if len(coords) < 2:
                raise ValueError(
                    f"Point '{point_id}' must contain at least [x, y]."
                )
            drawer.points[point_id] = Point2D(
                point_id,
                float(coords[0]),
                float(coords[1]),
            )

        for line_id, line in metadata.get("lines", {}).items():
            kind = line.get("kind", "side")
            drawer._check_line_kind(kind)
            drawer._check_point_exists(line["start"])
            drawer._check_point_exists(line["end"])
            drawer.lines[line_id] = Line2D(
                line_id,
                line["start"],
                line["end"],
                kind,
            )

        for surface_id, surface in metadata.get("surfaces", {}).items():
            vertices = list(surface["vertices"])
            for point_id in vertices:
                drawer._check_point_exists(point_id)
            drawer.surfaces[surface_id] = Surface2D(surface_id, vertices)

        drawer.hole_punches = list(metadata.get("hole_punches", []))
        drawer.surface_holes = {
            str(surface_id): [list(loop) for loop in loops]
            for surface_id, loops in metadata.get("surface_holes", {}).items()
        }

        from ..patterns.hexagon.metadata import load_hex_metadata
        load_hex_metadata(drawer, metadata)
        drawer._point_count = len(drawer.points)
        drawer._line_count = len(drawer.lines)
        drawer._surface_count = len(drawer.surfaces)

        return drawer

    @classmethod
    def from_json(cls, filename: str) -> "TwoDDrawer":
        """
        Load 2D metadata from JSON and build a TwoDDrawer.
        """
        with open(filename, "r", encoding="utf-8") as file:
            return cls.from_metadata(json.load(file))

    # ------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------

    def _new_point_id(self) -> str:
        pid = f"p{self._point_count}"
        self._point_count += 1
        return pid

    def _new_line_id(self) -> str:
        lid = f"l{self._line_count}"
        self._line_count += 1
        return lid

    def _new_surface_id(self) -> str:
        sid = f"s{self._surface_count}"
        self._surface_count += 1
        return sid

    # ------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------

    def _check_point_exists(self, point_id: str) -> None:
        if point_id not in self.points:
            raise ValueError(f"Point '{point_id}' does not exist.")

    def _check_line_kind(self, kind: str) -> None:
        if kind not in LINE_STYLES:
            raise ValueError(f"Invalid line kind '{kind}'. Allowed: {set(LINE_STYLES)}")

    @staticmethod
    def _canonical_line_key(start: str, end: str) -> Tuple[str, str]:
        """
        Return a direction-independent key for a line.

        Example:
            ("p0", "p1") and ("p1", "p0") both become ("p0", "p1").
        """
        return tuple(sorted((start, end)))

    @staticmethod
    def _canonical_surface_key(vertices: List[str]) -> Tuple[str, ...]:
        """
        Return an order-independent key for a surface.

        Example:
            ["p0", "p1", "p2"] and ["p2", "p0", "p1"]
            are treated as the same surface.
        """
        return tuple(sorted(vertices))

    @staticmethod
    def _resolve_overlapped_line_kind(
        existing_kind: LineKind,
        new_kind: LineKind,
    ) -> LineKind:
        """
        Decide the final line kind when a new line overlaps an existing line.

        Construction lines yield to every other kind, a crease replaces a
        side or rigid line, and otherwise the existing kind stays. A valley
        drawn over a mountain (or the reverse) is an error.
        """
        if existing_kind == new_kind or new_kind == "construction":
            return existing_kind
        if existing_kind == "construction":
            return new_kind

        creases = {"valley", "mountain"}
        if existing_kind in creases and new_kind in creases:
            raise ValueError(
                f"Conflicting crease assignment: existing line is "
                f"'{existing_kind}', but new line is '{new_kind}'."
            )
        return new_kind if new_kind in creases else existing_kind

    # ------------------------------------------------------------
    # Find existing objects
    # ------------------------------------------------------------

    def find_point_by_coordinate(
        self,
        x: float,
        y: float,
        tol: Optional[float] = None,
    ) -> Optional[str]:
        """
        Find an existing point with the same coordinate within tolerance.

        Args:
            x: x coordinate.
            y: y coordinate.
            tol: coordinate tolerance. If None, use self.point_tol.

        Returns:
            Existing point ID if found; otherwise None.
        """
        if tol is None:
            tol = self.point_tol

        if self._point_index is not None and tol == self._point_index[0]:
            bucket = self._point_bucket(x, y, tol)
            if bucket is not None:
                # Nearby cells include points across cell boundaries. Keep the
                # exact circular tolerance and the first inserted match, even
                # when candidates are visited in a different spatial order.
                buckets = self._point_index[1]
                best = None
                bx, by = bucket
                # A second neighboring cell covers floating-point rounding at
                # exact tolerance boundaries (e.g. -tiny versus +tol).
                for ix in range(bx - 2, bx + 3):
                    for iy in range(by - 2, by + 3):
                        for order, pid in buckets.get((ix, iy), ()):
                            if best is not None and order >= best[0]:
                                continue
                            point = self.points[pid]
                            if math.hypot(point.x - x, point.y - y) <= tol:
                                best = order, pid
                return None if best is None else best[1]

        for pid, point in self.points.items():
            distance = math.hypot(point.x - x, point.y - y)

            if distance <= tol:
                return pid

        return None

    def find_line_by_points(self, start: str, end: str) -> Optional[str]:
        """
        Find an existing line with the same two endpoints.

        The direction does not matter:
            start-end is considered the same as end-start.

        Returns:
            Existing line ID if found; otherwise None.
        """
        target_key = self._canonical_line_key(start, end)
        if self._geometry_indexes is not None:
            return self._geometry_indexes[0].get(target_key)

        for lid, line in self.lines.items():
            line_key = self._canonical_line_key(line.start, line.end)

            if line_key == target_key:
                return lid

        return None

    def find_surface_by_vertices(self, vertices: List[str]) -> Optional[str]:
        """
        Find an existing surface with the same vertex set.

        The vertex order does not matter.

        Returns:
            Existing surface ID if found; otherwise None.
        """
        target_key = self._canonical_surface_key(vertices)
        if self._geometry_indexes is not None:
            return self._geometry_indexes[1].get(target_key)

        for sid, surface in self.surfaces.items():
            surface_key = self._canonical_surface_key(surface.vertices)

            if surface_key == target_key:
                return sid

        return None

    # ------------------------------------------------------------
    # Add geometry
    # ------------------------------------------------------------

    def add_point(
        self,
        x: float,
        y: float,
        point_id: Optional[str] = None,
        merge_if_duplicate: bool = True,
    ) -> str:
        """
        Add one 2D point.

        Args:
            x: x coordinate.
            y: y coordinate.
            point_id: optional custom point ID.
            merge_if_duplicate:
                If True, return the existing point ID when another point
                with the same coordinate already exists.
                If False, raise an error when the coordinate is repeated.

        Returns:
            point_id
        """
        x = float(x)
        y = float(y)

        existing_id = self.find_point_by_coordinate(x, y)

        if existing_id is not None:
            if merge_if_duplicate:
                return existing_id

            raise ValueError(
                f"Point coordinate ({x}, {y}) already exists as point '{existing_id}'."
            )

        if point_id is None:
            point_id = self._new_point_id()

        if point_id in self.points:
            raise ValueError(f"Point id '{point_id}' already exists.")

        self.points[point_id] = Point2D(point_id, x, y)
        if self._point_index is not None:
            width, buckets = self._point_index
            bucket = self._point_bucket(x, y, width)
            if bucket is not None:
                buckets.setdefault(bucket, []).append((len(self.points) - 1, point_id))
        return point_id

    def add_line(
        self,
        start: str,
        end: str,
        kind: LineKind = "side",
        line_id: Optional[str] = None,
    ) -> str:
        """
        Add one line segment.

        If the same geometric line already exists, the line is merged instead
        of creating a duplicate.

        Args:
            start: start point ID.
            end: end point ID.
            kind:
                valley       -> valley crease
                mountain     -> mountain crease
                side         -> exterior boundary edge
                rigid        -> internal rigid line, no folding
                construction -> helper drawing line
            line_id: optional custom line ID.

        Returns:
            line_id
        """
        self._check_point_exists(start)
        self._check_point_exists(end)
        self._check_line_kind(kind)

        if start == end:
            raise ValueError("A line cannot start and end at the same point.")

        existing_line_id = self.find_line_by_points(start, end)

        if existing_line_id is not None:
            existing_line = self.lines[existing_line_id]
            existing_line.kind = self._resolve_overlapped_line_kind(
                existing_line.kind, kind,
            )
            return existing_line_id

        if line_id is None:
            line_id = self._new_line_id()

        if line_id in self.lines:
            raise ValueError(f"Line id '{line_id}' already exists.")

        self.lines[line_id] = Line2D(line_id, start, end, kind)
        if self._geometry_indexes is not None:
            self._geometry_indexes[0][self._canonical_line_key(start, end)] = line_id
        return line_id

    def add_surface(
        self,
        vertices: List[str],
        surface_id: Optional[str] = None,
        auto_boundary: bool = False,
        boundary_kind: LineKind = "side",
    ) -> str:
        """
        Add one polygonal rigid panel.

        If the same surface already exists, the surface is merged instead
        of creating a duplicate.

        Args:
            vertices:
                Ordered point IDs. Prefer counterclockwise order.
            surface_id:
                Optional custom surface ID.
            auto_boundary:
                If True, automatically creates boundary lines around the surface.
            boundary_kind:
                Line kind for automatically created boundary lines.

        Returns:
            surface_id
        """
        if len(vertices) < 3:
            raise ValueError("A surface requires at least 3 vertices.")

        if len(set(vertices)) != len(vertices):
            raise ValueError(
                f"Surface vertices contain duplicates: {vertices}"
            )

        for pid in vertices:
            self._check_point_exists(pid)

        self._check_line_kind(boundary_kind)

        existing_surface_id = self.find_surface_by_vertices(vertices)

        if existing_surface_id is not None:
            if auto_boundary:
                self._add_boundary_lines(vertices, boundary_kind)

            return existing_surface_id

        if surface_id is None:
            surface_id = self._new_surface_id()

        if surface_id in self.surfaces:
            raise ValueError(f"Surface id '{surface_id}' already exists.")

        self.surfaces[surface_id] = Surface2D(surface_id, list(vertices))
        if self._geometry_indexes is not None:
            self._geometry_indexes[1][self._canonical_surface_key(vertices)] = surface_id

        if auto_boundary:
            self._add_boundary_lines(vertices, boundary_kind)

        return surface_id

    def _add_boundary_lines(
        self,
        vertices: List[str],
        boundary_kind: LineKind = "side",
    ) -> None:
        """
        Add boundary lines around a polygonal surface.

        Existing overlapped lines are automatically merged by add_line().
        """
        n = len(vertices)

        for i in range(n):
            self.add_line(
                vertices[i],
                vertices[(i + 1) % n],
                kind=boundary_kind,
            )

    # ------------------------------------------------------------
    # Convenience geometry
    # ------------------------------------------------------------

    def add_triangle(
        self,
        p1: str,
        p2: str,
        p3: str,
        surface_id: Optional[str] = None,
        auto_boundary: bool = False,
    ) -> str:
        """
        Add a triangular rigid panel.
        """
        return self.add_surface(
            [p1, p2, p3],
            surface_id=surface_id,
            auto_boundary=auto_boundary,
        )

    def add_parallelogram(
        self,
        p1: str,
        p2: str,
        p3: str,
        p4: str,
        surface_id: Optional[str] = None,
        auto_boundary: bool = False,
    ) -> str:
        """
        Add a 4-vertex rigid panel.

        The code does not yet check whether it is geometrically a true parallelogram.
        It only stores it as a 4-vertex rigid surface.
        """
        return self.add_surface(
            [p1, p2, p3, p4],
            surface_id=surface_id,
            auto_boundary=auto_boundary,
        )

    def add_hole_punch(
        self,
        center: Tuple[float, float],
        diameter: float,
        segments: int = 32,
        hole_id: Optional[str] = None,
        surface_id: Optional[str] = None,
    ) -> str:
        """
        Add a polygonal circular hole-punch cut line.

        The cut is represented by side lines around a regular polygon so it
        remains available to the existing 2D and CAD exporters. The hole
        metadata is also retained for drawing and JSON round-tripping.
        Pass ``surface_id`` to attach the cut to a panel for folding,
        stacking, and export as an inner face boundary in 3D.
        """
        diameter = finite("diameter", diameter, positive=True)
        segments = integer("segments", segments, minimum=8)
        if surface_id is not None and surface_id not in self.surfaces:
            raise ValueError(f"Surface '{surface_id}' does not exist.")

        if hole_id is None:
            hole_id = f"hole_{len(self.hole_punches)}"
        if any(hole.get("id") == hole_id for hole in self.hole_punches):
            raise ValueError(f"Hole id '{hole_id}' already exists.")

        cx, cy = float(center[0]), float(center[1])
        radius = diameter / 2.0
        point_ids = [
            self.add_point(
                cx + radius * math.cos(2.0 * math.pi * index / segments),
                cy + radius * math.sin(2.0 * math.pi * index / segments),
                point_id=f"{hole_id}_p{index}",
            )
            for index in range(segments)
        ]
        self._add_boundary_lines(point_ids, "side")

        self.hole_punches.append(
            {
                "id": hole_id,
                "center": [cx, cy],
                "diameter": diameter,
                "segments": segments,
            }
        )
        if surface_id is not None:
            self.surface_holes.setdefault(surface_id, []).append(point_ids)
        return hole_id

    # ------------------------------------------------------------
    # Counters / summary
    # ------------------------------------------------------------

    def count_lines_by_kind(self) -> Dict[str, int]:
        """Count all line objects by kind, including kinds with no lines."""
        counts = dict.fromkeys(LINE_STYLES, 0)
        for line in self.lines.values():
            counts[line.kind] += 1
        return counts

    def count_creases(self) -> int:
        """
        Count physical crease segments.

        Only valley and mountain lines are counted as creases.
        Side, rigid, and construction lines are not counted.
        """
        counts = self.count_lines_by_kind()
        return counts["valley"] + counts["mountain"]

    def count_valley_creases(self) -> int:
        """
        Count valley crease segments only.
        """
        return self.count_lines_by_kind()["valley"]

    def count_mountain_creases(self) -> int:
        """
        Count mountain crease segments only.
        """
        return self.count_lines_by_kind()["mountain"]

    def summary(self) -> dict:
        """
        Return a compact summary of the current pattern.
        """
        line_counts = self.count_lines_by_kind()
        return {
            "num_points": len(self.points),
            "num_lines": len(self.lines),
            "num_surfaces": len(self.surfaces),
            "num_creases": line_counts["valley"] + line_counts["mountain"],
            "num_valley_creases": line_counts["valley"],
            "num_mountain_creases": line_counts["mountain"],
            "line_counts": line_counts,
        }

    def print_summary(self) -> None:
        """
        Print a readable summary of the current pattern.
        """
        s = self.summary()

        print("Pattern summary")
        print("----------------")
        print(f"Points:             {s['num_points']}")
        print(f"Lines:              {s['num_lines']}")
        print(f"Surfaces:           {s['num_surfaces']}")
        print(f"Creases:            {s['num_creases']}")
        print(f"  Valley creases:   {s['num_valley_creases']}")
        print(f"  Mountain creases: {s['num_mountain_creases']}")
        print("")
        print("Line counts:")
        for kind, count in s["line_counts"].items():
            print(f"  {kind}: {count}")

    # ------------------------------------------------------------
    # Query / export
    # ------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Convert the pattern into a JSON-friendly dictionary.
        """
        data = {
            "metadata": {
                "unit": self.unit,
                "point_tol": self.point_tol,
            },
            "points": {
                pid: [p.x, p.y]
                for pid, p in self.points.items()
            },
            "lines": {
                lid: {
                    "start": line.start,
                    "end": line.end,
                    "kind": line.kind,
                }
                for lid, line in self.lines.items()
            },
            "surfaces": {
                sid: {
                    "vertices": surface.vertices,
                }
                for sid, surface in self.surfaces.items()
            },
        }

        if hasattr(self, "hex_units"):
            data["hex_units"] = self.hex_units
            data["hex_creases"] = self.hex_creases

        if self.hole_punches:
            data["hole_punches"] = self.hole_punches

        if self.surface_holes:
            data["surface_holes"] = self.surface_holes

        return data

    def save_json(self, filename: str) -> None:
        """
        Save the current pattern as a JSON file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dxf_string(self, **options) -> str:
        """Convert the current pattern to an ASCII DXF string.

        ``options`` are the keyword options of :meth:`save_dxf`.
        """
        from ..io.dxf_export import dxf_string_from_metadata

        return dxf_string_from_metadata(self.to_dict(), **options)

    def save_dxf(self, filename: str, **options):
        """
        Save the current 2D pattern as a DXF file.

        Options (see :func:`origamicad.io.dxf_string_from_metadata`):
        ``include_creases=True``, ``crease_style="dashed"``,
        ``include_construction=False``, ``include_rigid=True``,
        ``include_side=True``, and ``connecting_dots=False``.

        Crease lines are exported on valley/mountain crease layers. Use
        crease_style="dashed" for a CAD linetype, or
        crease_style=["real dashed", dash_length, gap_length] for separate
        continuous LINE entities with gaps at both crease endpoints. Dash and
        gap lengths use the pattern's length unit. Use include_creases=False
        to export only cut/rigid geometry. Set connecting_dots=True to replace
        closed side boundaries with evenly distributed 0.6 mm blank gaps,
        leaving solid cut lines over the remaining approximately 99% of the
        boundary perimeter. DXF output uses a complete AutoCAD 2000 document
        that preserves and declares the pattern's length unit for AutoCAD and
        SolidWorks.
        """
        from ..io.dxf_export import save_dxf

        return save_dxf(self.to_dict(), filename, **options)

    # ------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------

    def draw(
        self,
        show_points: bool = False,
        show_point_ids: bool = False,
        show_line_ids: bool = False,
        show_surface_ids: bool = False,
        equal_axis: bool = True,
        figsize: Tuple[float, float] = (10, 10),
        save_fig: bool = False,
        save_path: str = "pattern.png",
        dpi: int = 300,
    ) -> None:
        """
        Draw the 2D pattern using matplotlib.

        Args:
            save_fig:
                If True, save the figure with plt.savefig().
            save_path:
                Output path used when save_fig is True.
            dpi:
                Image resolution used when save_fig is True.
        """
        fig, ax = plt.subplots(figsize=figsize)

        def xy(point_id):
            point = self.points[point_id]
            return point.x, point.y

        # Surfaces first: every surface is a rigid physical panel, tinted from
        # the color cycle. Panels, outlines, and each line kind are drawn as
        # one collection each instead of one artist per panel and line.
        polygons = [
            [xy(pid) for pid in surface.vertices] for surface in self.surfaces.values()
        ]
        if polygons:
            ax.add_collection(PolyCollection(
                polygons, alpha=0.15, edgecolors="none", zorder=1,
                facecolors=[f"C{index % 10}" for index in range(len(polygons))],
            ))
            ax.add_collection(LineCollection(
                [polygon + polygon[:1] for polygon in polygons],
                colors="black", linewidths=0.8, zorder=2,
            ))
        if show_surface_ids:
            for surface, polygon in zip(self.surfaces.values(), polygons):
                cx, cy = (sum(values) / len(polygon) for values in zip(*polygon))
                ax.text(cx, cy, surface.id, ha="center", va="center")

        # Mask the polygonal cut-line interiors so hole punches are visible
        # as holes in the 2D preview. Their side-line boundaries are drawn
        # below and remain available to CAD export.
        for hole in self.hole_punches:
            cx, cy = hole["center"]
            ax.add_patch(
                Circle(
                    (cx, cy),
                    hole["diameter"] / 2.0,
                    facecolor="white",
                    edgecolor="none",
                    zorder=1.5,
                )
            )

        # Draw lines
        segments = {}
        for line in self.lines.values():
            p0, p1 = xy(line.start), xy(line.end)
            segments.setdefault(line.kind, []).append((p0, p1))
            if show_line_ids:
                ax.text(0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), line.id, fontsize=8)
        for kind, kind_segments in segments.items():
            style = line_style(kind)
            ax.add_collection(LineCollection(
                kind_segments, colors=style["color"], linestyles=style["linestyle"],
                linewidths=style["linewidth"], zorder=2,
            ))

        # Draw points
        if show_points and self.points:
            ax.scatter(
                [point.x for point in self.points.values()],
                [point.y for point in self.points.values()],
                s=20, color="black",
            )
            if show_point_ids:
                for point in self.points.values():
                    ax.text(
                        point.x, point.y, f" {point.id}",
                        fontsize=8, ha="left", va="bottom",
                    )

        ax.autoscale_view()
        if equal_axis:
            ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(f"x [{self.unit}]")
        ax.set_ylabel(f"y [{self.unit}]")
        ax.grid(True, alpha=0.3)
        add_line_legend(ax, set(segments))

        if save_fig:
            plt.savefig(save_path, dpi=dpi)

        plt.show()
