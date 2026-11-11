from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .two_d_drawer import add_line_legend, line_style


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
        coordinates = {pid: self.point_array(pid) for pid in self.points}

        # All panels form one collection and every line kind another, so the
        # cost of a figure does not grow with one artist per panel and line.
        if show_surfaces and self.surfaces:
            paths, codes = [], []
            for surface_id, surface in self.surfaces.items():
                outline = [coordinates[pid] for pid in surface["vertices"]]
                outer_normal = np.cross(outline[1] - outline[0], outline[-1] - outline[0])
                loops = [outline]
                for hole in self.surface_holes.get(surface_id, []):
                    # Opposite winding makes inner contours transparent in
                    # the projected compound path, instead of filling them.
                    inner = [coordinates[pid] for pid in hole]
                    inner_normal = np.cross(inner[1] - inner[0], inner[-1] - inner[0])
                    if np.dot(inner_normal, outer_normal) > 0:
                        inner.reverse()
                    loops.append(inner)
                paths.append([point for loop in loops for point in loop + loop[:1]])
                codes.append([
                    code for loop in loops
                    for code in [Path.MOVETO] + [Path.LINETO] * (len(loop) - 1) + [Path.CLOSEPOLY]
                ])
                if show_surface_ids:
                    ax.text(
                        *np.mean(outline, axis=0), surface_id,
                        ha="center", va="center", fontsize=8,
                    )
            panels = Poly3DCollection(
                paths, alpha=0.18, facecolor="lightgray", edgecolor="black", linewidth=0.8,
            )
            panels.set_verts_and_codes(paths, codes)
            ax.add_collection3d(panels)

        segments = {}
        for line_id, line in self.lines.items():
            p0, p1 = coordinates[line["start"]], coordinates[line["end"]]
            segments.setdefault(line["kind"], []).append((p0, p1))
            if show_line_ids:
                ax.text(*(0.5 * (p0 + p1)), line_id, fontsize=8)
        for kind, kind_segments in segments.items():
            style = line_style(kind)
            ax.add_collection3d(Line3DCollection(
                kind_segments, colors=style["color"],
                linestyles=style["linestyle"], linewidths=style["linewidth"],
            ))

        if coordinates:
            xyz = np.array(list(coordinates.values()))
            if show_points:
                ax.scatter(*xyz.T, s=25, color="black", depthshade=False)
                if show_point_ids:
                    for point_id, (x, y, z) in coordinates.items():
                        ax.text(x, y, z, f" {point_id}", fontsize=8)
            if equal_axis:
                self._set_axes_equal_3d(ax)
            else:
                ax.auto_scale_xyz(*xyz.T)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=view[0], azim=view[1])
        ax.grid(True, alpha=0.3)
        add_line_legend(ax, set(segments))

        if save_fig:
            plt.savefig(save_path, dpi=dpi)

        plt.show()

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
