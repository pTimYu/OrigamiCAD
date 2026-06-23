from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CadVisualizationMixin:
    """Matplotlib drawing helpers for Cadder."""

    def draw(
        self,
        show_points: bool = True,
        show_point_ids: bool = True,
        show_line_ids: bool = False,
        show_surface_ids: bool = False,
        show_surfaces: bool = True,
        equal_axis: bool = True,
        figsize: Tuple[float, float] = (8, 7),
        view: Tuple[float, float] = (25, -60),
    ) -> None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if show_surfaces:
            for surface_id, surface in self.surfaces.items():
                coords = [self.point_array(pid) for pid in surface["vertices"]]
                ax.add_collection3d(
                    Poly3DCollection(
                        [coords],
                        alpha=0.18,
                        facecolor="lightgray",
                        edgecolor="black",
                        linewidth=0.8,
                    )
                )

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
