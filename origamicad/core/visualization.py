from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as MplLine2D
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CadVisualizationMixin:
    """Matplotlib drawing helpers for Cadder."""

    def draw(
        self,
        show_points: bool = False,
        show_point_ids: bool = False,
        show_line_ids: bool = False,
        show_surface_ids: bool = False,
        show_surfaces: bool = False,
        equal_axis: bool = True,
        figsize: Tuple[float, float] = (8, 7),
        view: Tuple[float, float] = (25, -60),
        save_fig: bool = False,
        save_path: str = "model.png",
        dpi: int = 300,
    ) -> None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if show_surfaces:
            for surface_id, surface in self.surfaces.items():
                coords = [self.point_array(pid) for pid in surface["vertices"]]
                collection = Poly3DCollection(
                    [coords],
                    alpha=0.18,
                    facecolor="lightgray",
                    edgecolor="black",
                    linewidth=0.8,
                )
                holes = self.surface_holes.get(surface_id, [])
                if holes:
                    # Opposite winding makes inner contours transparent in
                    # the projected compound path, instead of filling them.
                    outer_normal = np.cross(coords[1] - coords[0], coords[-1] - coords[0])
                    path_vertices = coords + coords[:1]
                    codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 1) + [Path.CLOSEPOLY]
                    for loop in holes:
                        inner = [self.point_array(pid) for pid in loop]
                        inner_normal = np.cross(inner[1] - inner[0], inner[-1] - inner[0])
                        if np.dot(inner_normal, outer_normal) > 0:
                            inner.reverse()
                        path_vertices.extend(inner + inner[:1])
                        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(inner) - 1) + [Path.CLOSEPOLY])
                    collection.set_verts_and_codes([path_vertices], [codes])
                ax.add_collection3d(collection)

                if show_surface_ids:
                    center = np.mean(np.array(coords), axis=0)
                    ax.text(*center, surface_id, ha="center", va="center", fontsize=8)

        for line_id, line in self.lines.items():
            p0 = self.point_array(line["start"])
            p1 = self.point_array(line["end"])
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                **self._line_style_3d(line["kind"]),
            )

            if show_line_ids:
                mid = 0.5 * (p0 + p1)
                ax.text(*mid, line_id, fontsize=8)

        if show_points:
            for point_id, point in self.points.items():
                ax.scatter(point.x, point.y, point.z, s=25, color="black")

                if show_point_ids:
                    ax.text(point.x, point.y, point.z, f" {point_id}", fontsize=8)

        if equal_axis:
            self._set_axes_equal_3d(ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=view[0], azim=view[1])
        ax.grid(True, alpha=0.3)
        self._add_line_legend_3d(ax)

        if save_fig:
            plt.savefig(save_path, dpi=dpi)

        plt.show()

    @staticmethod
    def _line_style_3d(kind: str) -> dict:
        styles = {
            "valley": {"color": "blue", "linestyle": "--", "linewidth": 1.8},
            "mountain": {"color": "red", "linestyle": "-.", "linewidth": 1.8},
            "side": {"color": "black", "linestyle": "-", "linewidth": 2.2},
            "rigid": {"color": "black", "linestyle": "-", "linewidth": 1.0},
            "construction": {"color": "gray", "linestyle": ":", "linewidth": 1.0},
        }

        try:
            return styles[kind]
        except KeyError:
            raise ValueError(f"Unknown line kind: {kind}") from None

    def _add_line_legend_3d(self, ax) -> None:
        labels = {
            "side": "Side",
            "valley": "Valley crease",
            "mountain": "Mountain crease",
            "rigid": "Rigid line",
            "construction": "Construction line",
        }
        kind_order = ("side", "valley", "mountain", "rigid", "construction")
        present_kinds = {line["kind"] for line in self.lines.values()}
        handles = [
            MplLine2D([0], [0], label=labels[kind], **self._line_style_3d(kind))
            for kind in kind_order
            if kind in present_kinds
        ]

        if handles:
            ax.legend(handles=handles, loc="best")

    def _set_axes_equal_3d(self, ax) -> None:
        coords = np.array([[p.x, p.y, p.z] for p in self.points.values()], dtype=float)

        if coords.size == 0:
            return

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        centers = 0.5 * (mins + maxs)
        radius = 0.5 * max(float(np.max(maxs - mins)), 1.0)

        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
