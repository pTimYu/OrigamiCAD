"""Type definitions shared by hexagon layout and kinematic operations."""

from typing import Literal, TypeAlias, TypedDict

Coordinate: TypeAlias = tuple[float, float]
PointID: TypeAlias = str
SurfaceID: TypeAlias = str
CreaseKind: TypeAlias = Literal["mountain", "valley"]
CreaseSide: TypeAlias = Literal["previous_quad", "current_quad"]


class TriangleKind(TypedDict):
    surface: SurfaceID
    kind: CreaseKind
    local_index: int
    unit: int


class LocalCrease(TypedDict):
    unit: int
    local_index: int
    edge: list[PointID]
    triangle: SurfaceID
    quad: SurfaceID
    kind: CreaseKind
    side: CreaseSide


class HexUnit(TypedDict):
    count: int
    mid: list[PointID]
    side: list[PointID]
    triangles: list[SurfaceID]
    parallelograms: list[SurfaceID]
    surfaces: list[SurfaceID]
    triangle_kinds: list[TriangleKind]
    local_creases: list[LocalCrease]
