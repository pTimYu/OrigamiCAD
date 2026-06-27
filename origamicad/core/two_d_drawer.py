# two_d_drawer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as MplLine2D
import math


LineKind = Literal["valley", "mountain", "side", "rigid", "construction"]


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

        self._point_count = 0
        self._line_count = 0
        self._surface_count = 0

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

        drawer.hex_units = metadata.get("hex_units", [])
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
        allowed = {"valley", "mountain", "side", "rigid", "construction"}
        if kind not in allowed:
            raise ValueError(f"Invalid line kind '{kind}'. Allowed: {allowed}")

    @staticmethod
    def _is_crease_kind(kind: LineKind) -> bool:
        return kind in {"valley", "mountain"}

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

    def _resolve_overlapped_line_kind(
        self,
        existing_kind: LineKind,
        new_kind: LineKind,
    ) -> LineKind:
        """
        Decide the final line kind when a new line overlaps an existing line.

        Rules:
            1. Same kind -> keep it.
            2. Existing construction -> replace by new kind.
            3. New construction -> keep existing kind.
            4. Valley vs mountain conflict -> raise error.
            5. Crease beats non-crease.
            6. Non-crease vs non-crease -> keep existing kind.
        """
        if existing_kind == new_kind:
            return existing_kind

        # Construction has the lowest priority.
        if existing_kind == "construction":
            return new_kind

        if new_kind == "construction":
            return existing_kind

        existing_is_crease = self._is_crease_kind(existing_kind)
        new_is_crease = self._is_crease_kind(new_kind)

        # Valley/mountain conflict.
        if existing_is_crease and new_is_crease and existing_kind != new_kind:
            raise ValueError(
                f"Conflicting crease assignment: existing line is "
                f"'{existing_kind}', but new line is '{new_kind}'."
            )

        # Crease type has higher priority than side/rigid/construction.
        if existing_is_crease and not new_is_crease:
            return existing_kind

        if new_is_crease and not existing_is_crease:
            return new_kind

        # Example: side vs rigid. Neither is a crease, so keep existing.
        return existing_kind

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
            resolved_kind = self._resolve_overlapped_line_kind(
                existing_line.kind,
                kind,
            )

            existing_line.kind = resolved_kind
            return existing_line_id

        if line_id is None:
            line_id = self._new_line_id()

        if line_id in self.lines:
            raise ValueError(f"Line id '{line_id}' already exists.")

        self.lines[line_id] = Line2D(line_id, start, end, kind)
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

    # ------------------------------------------------------------
    # Counters / summary
    # ------------------------------------------------------------

    def count_lines_by_kind(self) -> Dict[str, int]:
        """
        Count all line objects by their kind.

        Returns:
            Dictionary like:
            {
                "valley": 2,
                "mountain": 3,
                "side": 10,
                "rigid": 1,
                "construction": 0
            }
        """
        counts = {
            "valley": 0,
            "mountain": 0,
            "side": 0,
            "rigid": 0,
            "construction": 0,
        }

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
            "num_creases": self.count_creases(),
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

        return data

    def save_json(self, filename: str) -> None:
        """
        Save the current pattern as a JSON file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dxf_string(
        self,
        include_creases: bool = True,
        crease_style: Literal["solid", "dashed"] = "dashed",
        include_construction: bool = False,
        include_rigid: bool = True,
        include_side: bool = True,
    ) -> str:
        """
        Convert the current 2D metadata to an ASCII DXF string.
        """
        from ..io.dxf_export import dxf_string_from_metadata

        return dxf_string_from_metadata(
            self.to_dict(),
            include_creases=include_creases,
            crease_style=crease_style,
            include_construction=include_construction,
            include_rigid=include_rigid,
            include_side=include_side,
        )

    def save_dxf(
        self,
        filename: str,
        include_creases: bool = True,
        crease_style: Literal["solid", "dashed"] = "dashed",
        include_construction: bool = False,
        include_rigid: bool = True,
        include_side: bool = True,
    ):
        """
        Save the current 2D pattern as a DXF file.

        Crease lines are exported on valley/mountain crease layers. Use
        crease_style="dashed" for dashed laser-cutting crease lines, or
        include_creases=False to export only cut/rigid geometry.
        """
        from ..io.dxf_export import save_dxf

        return save_dxf(
            self.to_dict(),
            filename,
            include_creases=include_creases,
            crease_style=crease_style,
            include_construction=include_construction,
            include_rigid=include_rigid,
            include_side=include_side,
        )

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

        # Draw surfaces first
        for surface in self.surfaces.values():
            coords = [self.points[pid] for pid in surface.vertices]
            xs = [p.x for p in coords]
            ys = [p.y for p in coords]

            # Close polygon
            xs_closed = xs + [xs[0]]
            ys_closed = ys + [ys[0]]

            # Every surface is a rigid physical panel
            ax.fill(xs, ys, alpha=0.15)
            ax.plot(xs_closed, ys_closed, color="black", linewidth=0.8)

            if show_surface_ids:
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                ax.text(cx, cy, surface.id, ha="center", va="center")

        # Draw lines
        for line in self.lines.values():
            p0 = self.points[line.start]
            p1 = self.points[line.end]

            style = self._line_style(line.kind)
            ax.plot([p0.x, p1.x], [p0.y, p1.y], **style)

            if show_line_ids:
                mx = 0.5 * (p0.x + p1.x)
                my = 0.5 * (p0.y + p1.y)
                ax.text(mx, my, line.id, fontsize=8)

        # Draw points
        if show_points:
            for point in self.points.values():
                ax.scatter(point.x, point.y, s=20, color="black")

                if show_point_ids:
                    ax.text(
                        point.x,
                        point.y,
                        f" {point.id}",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )

        if equal_axis:
            ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(f"x [{self.unit}]")
        ax.set_ylabel(f"y [{self.unit}]")
        ax.grid(True, alpha=0.3)
        self._add_line_legend(ax)

        if save_fig:
            plt.savefig(save_path, dpi=dpi)

        plt.show()

    @staticmethod
    def _line_style(kind: LineKind) -> dict:
        """
        Visual convention:
            valley       -> blue dashed
            mountain     -> red dash-dot
            side         -> black solid thick
            rigid        -> black solid thin
            construction -> gray dotted
        """
        if kind == "valley":
            return {"color": "blue", "linestyle": "--", "linewidth": 1.8}

        if kind == "mountain":
            return {"color": "red", "linestyle": "-.", "linewidth": 1.8}

        if kind == "side":
            return {"color": "black", "linestyle": "-", "linewidth": 2.2}

        if kind == "rigid":
            return {"color": "black", "linestyle": "-", "linewidth": 1.0}

        if kind == "construction":
            return {"color": "gray", "linestyle": ":", "linewidth": 1.0}

        raise ValueError(f"Unknown line kind: {kind}")

    def _add_line_legend(self, ax) -> None:
        """
        Add a legend for the line kinds present in the pattern.
        """
        labels = {
            "side": "Side",
            "valley": "Valley crease",
            "mountain": "Mountain crease",
            "rigid": "Rigid line",
            "construction": "Construction line",
        }
        kind_order: Tuple[LineKind, ...] = (
            "side",
            "valley",
            "mountain",
            "rigid",
            "construction",
        )
        present_kinds = {line.kind for line in self.lines.values()}
        handles = [
            MplLine2D([0], [0], label=labels[kind], **self._line_style(kind))
            for kind in kind_order
            if kind in present_kinds
        ]

        if handles:
            ax.legend(handles=handles, loc="best")
