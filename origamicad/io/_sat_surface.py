"""Write exact polygonal ACIS sheet geometry through ezdxf."""

from pathlib import Path

import numpy as np
from ezdxf.acis import const, sat
from ezdxf.acis.entities import Body, Lump, Shell
from ezdxf.acis.hdr import AcisHeader
from ezdxf.acis.mesh import PolyhedronFaceBuilder
from ezdxf.math import Vec3
from ezdxf.render import MeshBuilder
from shapely.geometry import Polygon


_MILLIMETRES_PER_UNIT = {
    "m": 1000.0, "meter": 1000.0, "metre": 1000.0,
    "cm": 10.0, "centimeter": 10.0, "centimetre": 10.0,
    "mm": 1.0, "millimeter": 1.0, "millimetre": 1.0,
}
_PLANAR_TOLERANCE_MM = 1e-6


class _PreciseSatData(sat.SatDataExporter):
    def write_double(self, value: float) -> None:
        # The default formatter keeps six significant digits.
        self.data.append(f"{value:.17g}")


class _PreciseSatExporter(sat.SatExporter):
    def make_data_exporter(self, record: sat.SatEntity) -> sat.SatDataExporter:
        return _PreciseSatData(self, record.data)


def _face_normal(coordinates: np.ndarray, outer: list[int], holes: list[list[int]]) -> np.ndarray:
    """Check the polygon and its holes in the plane of the panel."""
    origin = coordinates[outer[0]]
    boundaries = [coordinates[loop] - origin for loop in (outer, *holes)]
    for boundary in boundaries:
        edges = np.roll(boundary, -1, axis=0) - boundary
        if np.any(np.linalg.norm(edges, axis=1) == 0):
            raise ValueError("SAT face boundary contains a zero-length edge.")

    outline = boundaries[0]
    normal = np.cross(outline, np.roll(outline, -1, axis=0)).sum(axis=0)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0:
        raise ValueError("SAT face has zero area.")
    normal /= magnitude
    if any(np.max(np.abs(boundary @ normal)) > _PLANAR_TOLERANCE_MM for boundary in boundaries):
        raise ValueError("SAT export requires planar panels and hole boundaries.")

    u_axis = outline[1] / np.linalg.norm(outline[1])
    basis = np.column_stack((u_axis, np.cross(normal, u_axis)))
    polygon = Polygon(outline @ basis, [boundary @ basis for boundary in boundaries[1:]])
    if not polygon.is_valid:
        raise ValueError("SAT face has an invalid outer boundary or hole.")
    return normal


def write_sat_surface(
    path: Path,
    coordinates: np.ndarray,
    outer_loops: list[list[int]],
    inner_loops: list[list[list[int]]],
    unit: str,
) -> Path:
    """Write one open sheet with shared edges and oppositely wound inner loops."""
    try:
        scale = _MILLIMETRES_PER_UNIT[unit.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"SAT export supports metre, centimetre, and millimetre coordinates; got {unit!r}."
        ) from exc
    points = coordinates * scale
    if not np.isfinite(points).all():
        raise ValueError("SAT vertices must have finite coordinates.")

    normals = [
        _face_normal(points, outer, holes)
        for outer, holes in zip(outer_loops, inner_loops)
    ]
    mesh = MeshBuilder()
    mesh.vertices = [Vec3(point) for point in points]
    mesh.faces = outer_loops
    builder = PolyhedronFaceBuilder(mesh)
    builder.double_sided = True
    faces = builder.acis_faces()
    if len(faces) != len(outer_loops):
        raise ValueError("SAT export could not construct every panel face.")

    shell = Shell()
    for face, normal, holes in zip(faces, normals, inner_loops):
        face.surface.normal = Vec3(normal)
        for hole in holes:
            face.append_loop(builder.make_loop(hole[::-1]))
        shell.append_face(face)
    lump = Lump()
    lump.append_shell(shell)
    body = Body()
    body.append_lump(lump)

    header = AcisHeader()
    header.version = const.DEFAULT_SAT_VERSION
    header.units_in_mm = 1.0
    exporter = _PreciseSatExporter(header)
    exporter.export(body)
    path.write_text("\n".join(exporter.dump_sat()) + "\n", encoding="ascii")
    return path
