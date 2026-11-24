from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import fsum

from shapely import STRtree
from shapely.affinity import rotate, scale, translate
from shapely.geometry import MultiPolygon, Polygon, box


_Point = tuple[float, float]
_Polygon = Polygon | MultiPolygon
_TOLERANCE = 1e-8
_MAX_CONTACT_VERTICES = 24


@dataclass(frozen=True)
class _Placement:
    pattern_index: int
    rotation: float
    translation: _Point
    geometry: _Polygon
    mirrored: bool = False


@dataclass(frozen=True, eq=False)
class _Orientation:
    pattern_index: int
    rotation: float
    offset: _Point
    geometry: _Polygon
    width: float
    height: float
    vertices: tuple[_Point, ...]
    mirrored: bool = False


@dataclass(frozen=True)
class _Sweep:
    x_first: bool
    from_right: bool = False
    from_top: bool = False

    def key(self, x: float, y: float, orientation: _Orientation) -> _Point:
        x = -(x + orientation.width) if self.from_right else x
        y = -(y + orientation.height) if self.from_top else y
        rounded = round(x, 8), round(y, 8)
        return rounded if self.x_first else rounded[::-1]


@dataclass
class _Candidates:
    orientation: _Orientation
    bounds: tuple[float, float, float, float]
    sweep: _Sweep
    queue: list[tuple[_Point, float, float]] = field(default_factory=list)
    seen: set[_Point] = field(default_factory=set)
    neighbors_added: int = 0

    def add(self, x: float, y: float) -> None:
        left, bottom, right, top = self.bounds
        if not (
            left - _TOLERANCE <= x <= right + _TOLERANCE
            and bottom - _TOLERANCE <= y <= top + _TOLERANCE
        ):
            return
        x = min(max(x, left), right)
        y = min(max(y, bottom), top)
        key = self.sweep.key(x, y, self.orientation)
        if key not in self.seen:
            self.seen.add(key)
            heapq.heappush(self.queue, (key, x, y))


def pack_sheet(
    outlines: Sequence[_Polygon],
    quantities: Sequence[int | None],
    width: float,
    height: float,
    *,
    margin: float,
    spacing: float,
    rotations: Sequence[float],
    allow_mirror: bool = False,
) -> tuple[_Placement, ...]:
    """Pack prepared millimetre silhouettes inside a rectangular sheet.

    Compare deterministic scans from different sheet corners, with alternate
    rotation priorities to expose different rows and columns. Try reflected
    copies when permitted and prefer the unreflected layout on ties.
    """
    active = [index for index, quantity in enumerate(quantities) if quantity != 0]
    if not active:
        return ()
    usable_width, usable_height = width - 2 * margin, height - 2 * margin
    orientations = [
        _make_orientations(
            outline, index, rotations, usable_width, usable_height, allow_mirror=allow_mirror,
        )
        if quantities[index] != 0 else []
        for index, outline in enumerate(outlines)
    ]
    available_area = usable_width * usable_height
    areas = [outline.area for outline in outlines]
    request_count = sum(quantity or 0 for quantity in quantities)
    has_fill = any(quantities[index] is None for index in active)

    @lru_cache(maxsize=256)
    def contacts(fixed: _Orientation, moving: _Orientation) -> tuple[_Point, ...]:
        return _contact_offsets(fixed, moving, spacing)

    best = ()
    best_score = (-1, 0.0, 0)
    unreflected = [[item for item in choices if not item.mirrored] for choices in orientations]
    orientation_groups = (
        (unreflected, orientations) if unreflected != orientations else (orientations,)
    )
    for group in orientation_groups:
        for priority, ordered, sweep in _sweep_strategies(group, active, areas):
            placements = _pack_direction(
                ordered, priority, quantities, width, height, margin, spacing, sweep, contacts,
            )
            count = sum(quantities[p.pattern_index] is not None for p in placements)
            area = fsum(areas[p.pattern_index] for p in placements)
            score = (count, area, len(placements))
            if score > best_score:
                best, best_score = placements, score
            remaining_area = available_area - area
            if count == request_count and (not has_fill or remaining_area <= _TOLERANCE):
                return best
            if len(active) == 1 and remaining_area < areas[active[0]] - _TOLERANCE:
                return best
    return best


def _sweep_strategies(
    orientations: list[list[_Orientation]], active: Sequence[int], areas: Sequence[float],
) -> Iterator[tuple[tuple[int, ...], list[list[_Orientation]], _Sweep]]:
    box_areas = [
        min((item.width * item.height for item in choices), default=0.0)
        for choices in orientations
    ]
    priorities = dict.fromkeys((
        tuple(sorted(active, key=lambda index: -box_areas[index])),
        tuple(sorted(active, key=lambda index: -areas[index])),
    ))
    base = (_Sweep(False), _Sweep(True))
    other_corners = (
        _Sweep(x_first, from_right, from_top)
        for from_right, from_top in ((True, False), (False, True), (True, True))
        for x_first in (False, True)
    )
    for priority, sweep in product(priorities, base):
        yield priority, orientations, sweep
    for sweep in (*base, *other_corners):
        for priority, ordered in product(priorities, _rotation_orders(orientations, sweep)):
            yield priority, ordered, sweep


def _rotation_orders(
    orientations: Sequence[list[_Orientation]], sweep: _Sweep,
) -> list[list[list[_Orientation]]]:
    # A column can start with a wide, short part if it leaves room for an
    # extra copy along the column. Apply the analogous preference in rows.
    def stack_size(item: _Orientation) -> float:
        return item.height if sweep.x_first else item.width

    compact = [sorted(choices, key=stack_size) for choices in orientations]
    broad = [sorted(choices, key=lambda item: -stack_size(item)) for choices in orientations]
    orders = []
    for order in (compact, broad):
        if order != orientations and order not in orders:
            orders.append(order)
    if sweep.from_right or sweep.from_top:
        orders.insert(0, list(orientations))
    return orders


def _make_orientations(
    outline: _Polygon,
    index: int,
    rotations: Sequence[float],
    width: float,
    height: float,
    *,
    allow_mirror: bool = False,
) -> list[_Orientation]:
    orientations = []
    normalized = []
    versions = [(outline, False)]
    if allow_mirror:
        versions.append((scale(outline, xfact=-1.0, origin=(0.0, 0.0)), True))
    for (source, mirrored), angle in product(versions, rotations):
        oriented = rotate(source, angle, origin=(0.0, 0.0))
        xmin, ymin, xmax, ymax = oriented.bounds
        size_x, size_y = xmax - xmin, ymax - ymin
        if size_x > width + _TOLERANCE or size_y > height + _TOLERANCE:
            continue
        geometry = translate(oriented, xoff=-xmin, yoff=-ymin)
        canonical = geometry.normalize()
        if any(canonical.equals_exact(other, _TOLERANCE) for other in normalized):
            continue
        normalized.append(canonical)
        orientations.append(_Orientation(
            index, angle, (-xmin, -ymin), geometry, size_x, size_y,
            _contact_vertices(geometry), mirrored,
        ))
    return orientations


def _contact_vertices(geometry: _Polygon) -> tuple[_Point, ...]:
    simplified = geometry.simplify(_TOLERANCE)
    polygons = (simplified,) if isinstance(simplified, Polygon) else simplified.geoms
    vertices = [point for polygon in polygons for point in polygon.exterior.coords[:-1]]
    if len(vertices) <= _MAX_CONTACT_VERTICES:
        return tuple(vertices)
    return tuple(
        vertices[index * len(vertices) // _MAX_CONTACT_VERTICES]
        for index in range(_MAX_CONTACT_VERTICES)
    )


def _contact_offsets(
    fixed: _Orientation, moving: _Orientation, spacing: float,
) -> tuple[_Point, ...]:
    """Return reusable translations that put a moving part beside a fixed one."""
    if spacing > 0:
        clearance = fixed.geometry.buffer(spacing, join_style="mitre", mitre_limit=2)
        fixed_vertices = _contact_vertices(clearance)
    else:
        fixed_vertices = fixed.vertices

    x_contacts = (
        -moving.width - spacing, fixed.width + spacing, 0.0, fixed.width - moving.width,
    )
    y_contacts = (
        -moving.height - spacing, fixed.height + spacing, 0.0, fixed.height - moving.height,
    )
    offsets = dict.fromkeys((x, y) for x in x_contacts for y in y_contacts)
    offsets.update(dict.fromkeys(
        (x - u, y - v) for x, y in fixed_vertices for u, v in moving.vertices
    ))
    return tuple(
        (x, y) for x, y in offsets
        if _clear_of(translate(moving.geometry, xoff=x, yoff=y), fixed.geometry, spacing)
    )


def _clear_of(candidate: _Polygon, fixed: _Polygon, spacing: float) -> bool:
    if spacing > 0:
        distance = candidate.distance(fixed)
        return distance > 0 and distance >= spacing - min(_TOLERANCE, spacing * 0.01)
    if candidate.relate_pattern(fixed, "F********"):
        return True
    # Rotated edges can overlap by a floating-point sliver at an exact contact.
    tolerance = min(
        _TOLERANCE * min(candidate.length, fixed.length),
        1e-8 * min(candidate.area, fixed.area),
    )
    return candidate.intersection(fixed).area <= tolerance


def _pack_direction(
    orientations: Sequence[list[_Orientation]],
    priority: Sequence[int],
    quantities: Sequence[int | None],
    width: float,
    height: float,
    margin: float,
    spacing: float,
    sweep: _Sweep,
    contacts: Callable[[_Orientation, _Orientation], tuple[_Point, ...]],
) -> tuple[_Placement, ...]:
    pools = []
    for pattern_orientations in orientations:
        candidates = []
        for orientation in pattern_orientations:
            right = max(margin, width - margin - orientation.width)
            top = max(margin, height - margin - orientation.height)
            pool = _Candidates(orientation, (margin, margin, right, top), sweep)
            for x, y in ((margin, margin), (right, margin), (margin, top), (right, top)):
                pool.add(x, y)
            candidates.append(pool)
        pools.append(candidates)

    placed = []
    neighbors = []
    occupied = []
    tree = STRtree(occupied)

    def place(index: int) -> bool:
        nonlocal tree
        choices = []
        for pool in pools[index]:
            for orientation, left, bottom in neighbors[pool.neighbors_added:]:
                for dx, dy in contacts(orientation, pool.orientation):
                    pool.add(left + dx, bottom + dy)
            pool.neighbors_added = len(neighbors)

            while pool.queue:
                position, x, y = pool.queue[0]
                candidate = translate(pool.orientation.geometry, xoff=x, yoff=y)
                region = candidate
                if spacing > 0:
                    xmin, ymin, xmax, ymax = candidate.bounds
                    region = box(xmin - spacing, ymin - spacing, xmax + spacing, ymax + spacing)
                if all(_clear_of(candidate, occupied[i], spacing) for i in tree.query(region)):
                    choices.append((position, pool, x, y, candidate))
                    break
                heapq.heappop(pool.queue)

        if not choices:
            return False
        _, pool, x, y, geometry = min(choices, key=lambda choice: choice[0])
        orientation = pool.orientation
        offset_x, offset_y = orientation.offset
        placed.append(_Placement(
            index, orientation.rotation, (x + offset_x, y + offset_y), geometry,
            orientation.mirrored,
        ))
        neighbors.append((orientation, x, y))
        occupied.append(geometry)
        tree = STRtree(occupied)
        heapq.heappop(pool.queue)
        return True

    unlimited = []
    for index in priority:
        quantity = quantities[index]
        if quantity is None:
            unlimited.append(index)
        else:
            for _ in range(quantity):
                if not place(index):
                    break

    # Each unlimited variant gets one attempt per pass to preserve the mix.
    while unlimited:
        before = len(placed)
        for index in unlimited:
            place(index)
        if len(placed) == before:
            break
    return tuple(placed)
